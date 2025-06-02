import os
import pickle
import orbax.checkpoint
from flax.training import orbax_utils
import jax
import jax.numpy as jnp
import numpy as np

# suppress logging from orbax 
import logging
logger = logging.getLogger("absl")
logger.setLevel(logging.ERROR)

# compute path to repo root by using this file's path
REPO_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def save_train_run(out, savedir, savename):
    '''Save train run as orbax checkpoint. 
    Orbax requires absolute paths, so we compute the absolute path to the repo root.'''
    # determine whether savedir is relative or absolute
    if not os.path.isabs(savedir):
        savedir = os.path.join(REPO_PATH, savedir)
    if not os.path.exists(savedir):
        os.makedirs(savedir, exist_ok=True)
    savepath = os.path.join(savedir, savename)
    
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(out)
    
    # Save the checkpoint
    checkpointer.save(savepath, out, save_args=save_args)
    return savepath

def load_checkpoints(path, ckpt_key="checkpoints", custom_loader_cfg: dict=None):
    '''Load checkpoints from orbax checkpoint. 
    Orbax requires absolute paths, so we compute the absolute path to the repo root.'''
    restored = load_train_run(path)
    if custom_loader_cfg is None:
        return restored[ckpt_key]
    elif custom_loader_cfg["name"] == "open_ended":
        partner_out, ego_out = restored
        out = ego_out if custom_loader_cfg["type"] == "ego" else partner_out
        if ckpt_key == "final_buffer":
            return out["final_buffer"]["params"]
        else:
            return out[ckpt_key]
    else:
        raise ValueError(f"Invalid custom loader name: {custom_loader_cfg['name']}")

def load_train_run(path):
    '''Load checkpoints from orbax checkpoint. 
    Orbax requires absolute paths, so we compute the absolute path to the repo root.'''
    # determine whether path is relative or absolute
    if not os.path.isabs(path):
        path = os.path.join(REPO_PATH, path)
    # load the checkpoint
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    restored = checkpointer.restore(path)
    # convert pytree leaves from np arrays to jax arrays
    restored = jax.tree_util.tree_map(
        lambda x: jnp.array(x) if isinstance(x, np.ndarray) else x,
        restored
    )
    return restored

def save_train_run_as_pickle(out, savedir, savename):
    if not os.path.exists(savedir):
        os.makedirs(savedir, exist_ok=True)
        
    savepath = f"{savedir}/{savename}.pkl"
    with open(savepath, "wb") as f:
        pickle.dump(out, f)
    return savepath

def load_checkpoints_from_pickle(path, ckpt_key="checkpoints"):
    out = load_train_run_from_pickle(path)
    return out[ckpt_key]

def load_train_run_from_pickle(path):
    with open(path, "rb") as f:
        out = pickle.load(f)
    return out