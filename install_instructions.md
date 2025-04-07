### Instructions
Instructions were tested on debruyne by Caroline on 4/3/25 on a fresh install w/Python 3.11. 

1. Create a conda env: 

Command to install in scratch space with prefix:
 ```conda create --prefix /scratch/cluster/<username>/conda_envs/<your_env_name> python=3.11```

Command to install in default conda env location: 
```conda create --name your_env_name python=3.11```

2. Activate your conda environment via 
```conda activate your_env_name```

3. Install packages via; 
```pip install -r requirements.txt```

4. Verify that cuda is available via running `import jax; jax.devices()` in the Python interpreter.
You should see something like the following output: 

```
[CudaDevice(id=0), CudaDevice(id=1), CudaDevice(id=2), CudaDevice(id=3)]
```

If you instead see a warning message that a CPU-only version of jax was installed, manually run: 
```pip install --upgrade "jax[cuda12]"```

5. While your conda environment is activated, add project path to the PYTHONPATH as a conda env var:

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
```python ppo/ippo.py```
### Instructions
Instructions were tested on debruyne by Caroline on 4/3/25 on a fresh install w/Python 3.11. 

1. Create a conda env: 

Command to install in scratch space with prefix:
 ```conda create --prefix /scratch/cluster/<your_username>/conda_envs/<your_env_name> python=3.11```

Command to install in default conda env location: 
```conda create --name your_env_name python=3.11```

2. Activate your conda environment via 
```conda activate your_env_name```

3. Install packages via; 
```pip install -r requirements.txt```

4. Verify that cuda is available via running `import jax; jax.devices()` in the Python interpreter.
You should see something like the following output: 

```
[CudaDevice(id=0), CudaDevice(id=1), CudaDevice(id=2), CudaDevice(id=3)]
```
If you instead see a warning message that a CPU-only version of jax was installed, manually run: 
```pip install --upgrade "jax[cuda12]"```

5. While your conda environment is activated, add project path to the PYTHONPATH as a conda env var:

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
```python ppo/ippo.py```
