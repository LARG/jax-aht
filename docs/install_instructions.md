# Instructions
These instructions target Python 3.11.

1. Create a conda environment: 

Command to install in specific directory with prefix:
```bash
conda create --prefix <path_of_choice>/<your_env_name> python=3.11
```

Command to install in default conda env location: 
```bash
conda create --name your_env_name python=3.11
```

2. Activate your conda environment:
```bash
conda activate your_env_name
```

3. Navigate to the repository directory and install in development mode:
```bash
cd /path/to/jax-aht
python -m pip install -e .
```

This installs the CPU-compatible dependency set from `pyproject.toml` and sets up the package for development.

On Linux machines with NVIDIA GPUs and CUDA 12, install the CUDA extra instead:
```bash
python -m pip install -e ".[cuda12]"
```

4. Verify the installation:
```bash
python scripts/verify_install.py
```

This reports the devices JAX can see and runs the array operations the training
code relies on. The output should end with `All checks passed.`. A default CPU
installation looks like:
```
jax 0.5.3, backend cpu, [CpuDevice(id=0)]
[ok] devices
[ok] matrix multiplication
[ok] QR decomposition
[ok] orthogonal init (as used by agents/)
All checks passed.
```
With the CUDA extra installed, the backend should be `gpu` and the devices should
be listed as `CudaDevice` entries.

5. Download evaluation data to get the evaluation agents:
```bash
python download_eval_data.py
```

6. Test the installation by running our IPPO implementation: 
```bash
python marl/run.py task=lbf/lbf_7x7_nolevels algorithm=ippo/lbf/lbf_7x7_nolevels
```


# Alternative Manual Installation

If you prefer the manual setup or encounter issues with the pip installation:

1. Follow steps 1-2 above
2. Install packages manually: `python -m pip install -r requirements.txt`

   On Linux machines with NVIDIA GPUs and CUDA 12, also run `python -m pip install "jax[cuda12]==0.5.3"`.
3. Add project path to PYTHONPATH as a conda env var:
```bash
conda env config vars set PYTHONPATH=/path/to/repository/directory

# deactivate and reactivate to apply changes
conda deactivate 
conda activate your_env_name

# verify that pythonpath has been modified to include the current project dir
echo $PYTHONPATH
```

*If for some reason you need to remove the conda env var, you can run:
```bash
conda env config vars unset PYTHONPATH
```
4. Follow remaining installation steps from Step 4 onwards. 

# Troubleshooting

We provide some basic troubleshooting guidance.

## If JAX is not importable after installation

Confirm that `pip` and `python` point to the same environment:
```
which python
which pip
python -m pip --version
```

If they do not point to the same environment, reinstall using:
```
python -m pip install -e .
```

If `jumanji` fails with `ModuleNotFoundError: No module named 'pkg_resources'`, your environment likely has `setuptools>=81`, which removed `pkg_resources`. Reinstall a compatible version:
```
python -m pip install "setuptools<81" --force-reinstall
python -m pip install -e .
```

## CUDA library conflicts

If you installed the `cuda12` extra, pip installed its own CUDA 12 libraries. If a
system CUDA toolkit is also on the library search path, the two can be mixed at
runtime, which causes segfaults or cuBLAS/cuSolver errors after `jax.devices()`
already reports a GPU. Check with `echo $LD_LIBRARY_PATH`.
If the output is not empty, use:
```bash
export LD_LIBRARY_PATH="" #so that it defaults to the pip-installed CUDA.
conda env config vars set LD_LIBRARY_PATH= #so that it unsets the LD_LIBRARY_PATH when the conda environment is activated.
```

On macOS, CUDA is not used. If you see CPU devices from `jax.devices()`, that is the expected behavior.

## If video export fails

Some evaluation and test scripts save `.mp4` files through the `ffmpeg` executable. Install it separately if needed.

On macOS:
```
brew install ffmpeg
```

On Ubuntu/Debian:
```
sudo apt-get update
sudo apt-get install -y ffmpeg
```

Verify it is available with:
```
ffmpeg -version
```
