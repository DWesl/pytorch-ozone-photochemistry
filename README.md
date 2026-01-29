# PyTorch ozone photochemistry

Create a stratospheric ozone photochemistry physics parametrization scheme using a neural net 
trained on dynamical data.

Edit `mamba-env-create.sh` and run to install the software needed.  Edit `pytorch_simple.sh`
to match the changes made in `mamba-env-create.sh` and run or submit.

Mamba and conda are nearly interchangable; if the former is not installed, change all invocations
of mamba to invocations of conda, and change the `mamba run -p ... \` line in `pytorch_simple.sh`
to a `conda activate -p ...` line prior to that point.

The model definition and training code is in `pytorch_model.py`.
