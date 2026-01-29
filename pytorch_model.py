#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fit a pytorch model to ozone increments."""
import contextlib
import logging
import os
import shutil
from typing import Callable, Optional

import dask
import dask.array as da
import numpy as np
import pandas as pd
import torch.cuda
import torch.distributed
import torch.optim
import xarray as xr
from torch import nn
from torch.utils.data import DataLoader, Dataset, DistributedSampler

_LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level="INFO", format="%(asctime)s:%(levelname)s:%(name)s:%(message)s"
)

# Parallelism already happens at the PyTorch layer
# Don't try to add dask parallelization on top of that.
dask.config.set(scheduler="synchronous")


np.set_printoptions(linewidth=shutil.get_terminal_size().columns)
xr.set_options(display_width=shutil.get_terminal_size().columns)

SLURM_CPUS_PER_TASK = int(os.environ.get("SLURM_CPUS_PER_TASK", 2))


def totensor(arr: da.Array) -> torch.Tensor:
    if isinstance(arr, pd.DataFrame):
        arr = arr.values
    elif isinstance(arr, pd.Series):
        arr = arr.values[:, np.newaxis]
    return torch.Tensor(arr).float()


class NCFileDataset(Dataset):
    """Organize data from netCDF files."""

    def __init__(
        self,
        data_glob: str = "regression-data/??/all/gdas.f06+inc.202[0-4]??.nc",
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        target_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        is_train=True,
    ):
        self.ds = xr.open_mfdataset(
            data_glob,
            preprocess=lambda ds: (
                ds[["o3mr_net_prod", "o3mr", "t", "o3mr_part_col"]]
                .drop_vars(["time", "step", "cell_areas"])
                .sel(
                    isobaricInhPa=[
                        # fmt: off
                        400, 350, 300, 250, 200, 150, 100,
                        70, 50, 30, 20, 10,
                        7, 5, 3, 2, 1
                        # fmt: on
                    ],
                    latitude=slice(None, None, 5),
                    longitude=slice(None, None, 5),
                )
                .load()
            ),
        )
        if is_train:
            # Random-access from disk is slow, so load the datasets
            # that'll be shuffled.  Sequential access should be
            # decently fast, so can re-read from disk when needed
            self.ds = self.ds.persist()
        self.ds.coords["isobaricInPa"] = (
            ("isobaricInhPa",),
            self.ds.coords["isobaricInhPa"].values * 100,
            self.ds.coords["isobaricInhPa"].attrs | {"units": "Pa"},
        )
        self.ds.coords["doy"] = self.ds.coords["valid_time"].dt.dayofyear
        # _LOGGER.info(
        #     "Dataset counts and sizes:\n%s\n%s",
        #     self.ds.count(["valid_time", "latitude", "longitude"]),
        #     self.ds.sizes,
        # )
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return self.ds["o3mr_net_prod"].size

    def __getitem__(self, flat_index: int) -> tuple:
        ds = self.ds
        dimension_indexes = np.unravel_index(flat_index, ds["t"].shape)
        # _LOGGER.info(
        #     "Retrieving item %d/%s from\n%s", flat_index, dimension_indexes, ds
        # )
        data = ds.isel(
            {dim: idx for dim, idx in zip(ds.dims, dimension_indexes)}
        ).expand_dims("index")
        data = data.to_dataframe().drop(
            ["isobaricInhPa", "longitude", "valid_time"], axis=1
        )
        # assert np.isfinite(data).all().all()
        regressors = totensor(
            data[["o3mr", "t", "o3mr_part_col", "doy", "isobaricInPa", "latitude"]]
        )
        if self.transform is not None:
            regressors = self.transform(regressors)
        target = totensor(data["o3mr_net_prod"])
        if self.target_transform is not None:
            target = self.target_transform(target)
        return (regressors, target)

    def __getitems__(self, flat_indexes: list[int]) -> tuple:
        ds = self.ds
        dimension_indexes = np.unravel_index(flat_indexes, ds["t"].shape)
        data = (
            ds.isel(
                {
                    dim: xr.DataArray(idx, dims=("index",))
                    for dim, idx in zip(ds.dims, dimension_indexes)
                }
            )
            .to_dataframe()
            .drop(["isobaricInhPa", "longitude", "valid_time"], axis=1)
        )
        regressors = totensor(
            data[["o3mr", "t", "o3mr_part_col", "doy", "isobaricInPa", "latitude"]]
        )
        if self.transform is not None:
            regressors = self.transform(regressors)
        target = totensor(data[["o3mr_net_prod"]])
        if self.target_transform is not None:
            target = self.target_transform(target)
        return (regressors, target)


def train_iteration(
    loader: DataLoader,
    model: nn.Module,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optim,
):
    """Train the model, running through all the data once."""
    size = len(loader.dataset)
    model.train()
    for batch, (X, y) in enumerate(loader):
        X = X.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        optim.zero_grad()
        predicted = model(X)
        loss = loss_fn(predicted, y)

        loss.backward()
        optim.step()
        optim.zero_grad()
        if batch % 1000 == 0:
            curr_loss, curr_count = loss.item(), batch * train_batch_size + len(X)
            _LOGGER.info(f"loss: {curr_loss:>7.5g}  [{curr_count:>5d}/{size:>5d}]")


def test_iteration(
    loader: DataLoader,
    model: nn.Module,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
):
    """Check the model against the test data."""
    # size = len(loader.dataset)
    model.eval()

    num_batches = len(loader)
    assert np.isfinite(num_batches)
    test_loss = 0

    with torch.no_grad():
        for X, y in loader:
            X = X.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            pred = model(X)
            here_loss = loss_fn(pred, y).item()
            if np.isfinite(here_loss):
                test_loss += here_loss

    test_loss /= num_batches
    _LOGGER.info("Test MSE: %8g\n", test_loss)


@contextlib.contextmanager
def pytorch_distributed():
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    try:
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        _LOGGER.info(
            "Process initialized - Rank %d, World size: %d, local rank: %d, Device: cuda:%d",
            rank,
            world_size,
            local_rank,
            local_rank,
        )
        torch.cuda.set_device(local_rank)
        yield rank, world_size, local_rank
    finally:
        torch.distributed.destroy_process_group()


def save_checkpoint(
    model: nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    path: str,
    mlflow_enabled: bool = False,
):
    if torch.distributed.get_rank() == 0:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
            },
            path,
        )
        _LOGGER.info("[Rank 0] Saved checkpoint at epoch %d to %s", epoch, path)
        log_mlflow_artifact(
            path, artifact_path="mlflow-artifacts", mlflow_enabled=mlflow_enabled
        )


def load_checkpoint(model: nn.Module, optimizer, scheduler, path: str) -> int:
    _LOGGER.info(
        "[Rank %d] Loading checkpoint from %s", torch.distributed.get_rank(), path
    )
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    try:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    except KeyError:
        pass
    return checkpoint["epoch"]


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 64,
    num_workers: int = SLURM_CPUS_PER_TASK,
    is_train: bool = True,
):
    sampler = DistributedSampler(dataset, shuffle=is_train)
    # sampler = None
    # The sampler supersedes the DataLoader's shuffle parameter,
    # raising an error if both are present.
    return (
        DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            # My dataset returns a tensor already, which is what
            # collate_fn ensures
            collate_fn=lambda arr: arr,
        ),
        sampler,
    )


def log_mlflow_artifact(path: str, artifact_path=None, mlflow_enabled=False):
    if mlflow_enabled:
        import mlflow

        try:
            mlflow.log_artifact(path, artifact_path=artifact_path)
        except Exception as err:
            _LOGGER.warn("Failed to log artifact %s\n%s", path, err)


class SirenActivation(nn.Module):
    def __init__(self, omega):
        super().__init__()
        self.omega = nn.Parameter(torch.Tensor([omega]))

    def forward(self, x):
        return torch.sin(self.omega * x)


class SirenNN(nn.Module):
    def __init__(self, hidden_size=768, n_hidden_layers=6):
        super().__init__()
        layers = [nn.Linear(6, hidden_size), SirenActivation(6.0)]
        subsequent_activation = SirenActivation(4.0)
        for _i in range(n_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(subsequent_activation)
        layers.append(nn.Linear(hidden_size, 1))
        self.network = nn.Sequential(*layers)
        self.convert_input = torch.Tensor([1e6, 1e-1, 1e2, 1e-1, 1e-2, 1e-1])
        self.convert_output = torch.Tensor([1e-12])

    def forward(self, input):
        return self.convert_output * self.network.forward(self.convert_input * input)

    def cuda(self):
        super().cuda()
        self.convert_input = self.convert_input.cuda()
        self.convert_output = self.convert_output.cuda()


if __name__ == "__main__":
    _LOGGER.info("Imports done, defining model")
    # Currently ~150min/epoch/month for just training
    # Sped up with __getitems__
    # MSE for a constant predictor of 0 is 1.13e-23

    with pytorch_distributed() as (rank, world_size, local_rank):

        train_batch_size = 256
        train_dataset = NCFileDataset(
            "regression-data/??/all/gdas.f06+inc.202[0-4]??.nc", is_train=True
        )
        # train_dataset = NCFileDataset("regression-data/01/all/gdas.f06+inc.202001.nc")
        train_loader, train_sampler = create_dataloader(
            train_dataset, batch_size=train_batch_size, is_train=True
        )

        test_batch_size = 360
        test_dataset = NCFileDataset(
            "regression-data/??/all/gdas.f06+inc.20[12][95]??.nc", is_train=False
        )
        # test_dataset = NCFileDataset("regression-data/01/all/gdas.f06+inc.202101.nc")
        test_loader, test_sampler = create_dataloader(
            test_dataset, batch_size=test_batch_size, is_train=False
        )

        # simple_model = nn.Sequential(nn.Linear(6, 10), nn.ReLU(), nn.Linear(10, 1))
        # simple_model.cuda()
        # intermediate_model = nn.Sequential(
        #     nn.Linear(6, 10), nn.ReLU(), nn.Linear(10, 7), nn.ReLU(), nn.Linear(7, 1)
        # )
        # intermediate_model.cuda()
        # siren_model = nn.Sequential(
        #     nn.Linear(6, 16),
        #     SirenActivation(16),
        #     nn.Linear(16, 16),
        #     SirenActivation(16),
        #     nn.Linear(16, 16),
        #     SirenActivation(16),
        #     nn.Linear(16, 1),
        # )
        siren_model = SirenNN()
        siren_model.cuda()
        loss_fn = nn.MSELoss()

        # 2.5e-5/0.4 LR works okay for simple_model (3e-5 might work better)
        # For intermediate_model: 5e-5/0.4 gets to ~constant
        # For three-layer model: 5e-5/0.5 overfits
        # For siren model: 5e-5 only gets to 1e-5 loss
        # With converted units: 6e-4/0.6/5 starts decent and doesn't change
        # 5e-5/0.5/5 doesn't change much
        learning_rate = 2e-5
        optimizer = torch.optim.SGD(siren_model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.75)
        _LOGGER.info("Learning rate: %.1e\nScheduler: %s", learning_rate, scheduler)

        # Initial metric values, as a baseline
        test_iteration(test_loader, siren_model, loss_fn, optimizer)

        _LOGGER.info("Model defined, starting training\n%s\n", siren_model)
        epochs = 5
        for t in range(epochs):
            _LOGGER.info(f"\nEpoch {t+1:d}/{epochs}\n---------")
            if train_sampler is not None:
                train_sampler.set_epoch(t + 1)
            train_iteration(train_loader, siren_model, loss_fn, optimizer)
            test_iteration(test_loader, siren_model, loss_fn)
            scheduler.step()

            _LOGGER.info("Saving checkpoint")
            save_checkpoint(
                siren_model,
                optimizer,
                scheduler,
                t,
                f"siren_model_checkpoint-{t:d}.chkpt",
            )

    torch.save(siren_model.state_dict(), "siren_model_weights.pt1")
    torch.save(siren_model, "siren_model.pt0")

    # TODO: Check whether this needs to be Dim.DYNAMIC("batch")
    batch_dim = torch.export.Dim("batch")
    torch.export.save(
        torch.export.export(
            siren_model, (torch.randn(10, 6).cuda(),), dynamic_shapes=[{0: batch_dim}]
        ),
        "siren_model_exported.pt2",
    )
