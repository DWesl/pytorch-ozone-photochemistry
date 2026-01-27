#!/bin/bash

# Tell mamba/conda what version of CUDA we have
export CONDA_OVERRIDE_CUDA=12.9.0

# I was confused by the directions for license compliance with conda,
# so I installed and use mamba.  They are mostly interchangable
mamba create -p /scratch3/NCEPDEV/global/${USER}/AIML/conda/env \
      -r /scratch3/NCEPDEV/global/${USER}/AIML/conda/root -c conda-forge \
      gputil numpy psutil pytorch-gpu scikit-learn xarray pandas dask netCDF4

