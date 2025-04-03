### Instructions

1. Update conda via `conda update conda` and update pip via `pip install --upgrade pip`

2. Create a conda env: 

Command to install in scratch space with prefix:
 ```conda create --prefix /scratch/cluster/clw4542/conda_envs/your_env_name python==3.11```

Command to install in default conda env location: 
```conda create --name your_env_name python=3.11```

3. Activate your conda environment via 
```conda activate your_env_name```

4. Install packages via; 
```pip install -r requirements.txt```

5. Verify that cuda is available via running `import jax; jax.devices()` in the Python interpreter.
You should see something like the following output: 

```
[CudaDevice(id=0), CudaDevice(id=1), CudaDevice(id=2), CudaDevice(id=3)]
```

6. While your conda environment is activated, add project path to the PYTHONPATH as a conda env var:

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

7. Check if you can run our IPPO implementation: 
```python ppo/ippo.py```

### Notes: 
- Instructions were tested on debruyne by Caroline on 4/3/25 on a fresh install w/Python 3.11. 

Potential Issues: 
- Jax must use Cuda 12.2. However, the base cuda is Cuda 11.2. Check your cuda version with ```nvcc --version``` 
- If jax has auto-installed a CPU version, try `pip install -U jax[cuda-12]`