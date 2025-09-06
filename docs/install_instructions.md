### Instructions
Instructions were tested on 9/4/25 with a fresh install w/Python 3.11 on a server with Nvidia A100 GPUs.

1. Create a conda environment: 

Command to install in specific directory with prefix:
 ```conda create --prefix <path_of_choice>/<your_env_name> python=3.11```

Command to install in default conda env location: 
```conda create --name your_env_name python=3.11```

2. Activate your conda environment:
```conda activate your_env_name```

3. Navigate to the repository directory and install in development mode:
```
conda env config vars set PYTHONPATH=/path/to/repository/directory
cd /path/to/jax-aht
pip install -e .
```

This will automatically install all dependencies from `pyproject.toml` and set up the package for development.

4. Verify that CUDA is available by running `import jax; jax.devices()` in the Python interpreter.
You should see something like the following output:env--list
```
[CudaDevice(id=0)]
```

*If you instead see a warning message that a CPU-only version of Jax was installed, manually run: 
```pip install --upgrade "jax[cuda12]"```

*If you have cuda library installed elsewhere, check with `echo $LD_LIBRARY_PATH`. 
if the output is not empty use
```
export LD_LIBRARY_PATH="" #so that it defaults to the pip installed cuda.
conda env config vars set LD_LIBRARY_PATH= #so that it unset the LD_LIBRARY_PATH when conda env activated
```


5. Download evaluation data (optional, required for reproducing paper results):
```python download_eval_data.py```

6. Test the installation by running our IPPO implementation: 
```python marl/run.py task=lbf algorithm=ippo/lbf```

### Alternative Installation (Manual Method)

If you prefer the manual setup or encounter issues with the pip installation:

1. Follow steps 1-2 above
2. Install packages manually: `pip install -r requirements.txt`
3. Add project path to PYTHONPATH as a conda env var:
```
conda env config vars set PYTHONPATH=/path/to/repository/directory

# deactivate and reactivate to apply changes
conda deactivate 
conda activate your_env_name

# verify that pythonpath has been modified to include the current project dir
echo $PYTHONPATH
```

*if for some reason you need to remove the conda env var, you can run 
```conda env config vars unset PYTHONPATH```

6. Check if you can run our IPPO implementation: 
```python marl/run.py task=lbf algorithm=ippo/lbf```

