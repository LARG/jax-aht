import copy

import jaxmarl
import jumanji
from jumanji.environments.routing.lbf.generator import RandomGenerator as LbfGenerator

from envs.lbf.adhoc_lbf_viewer import AdHocLBFViewer
from envs.jumanji_jaxmarl_wrapper import JumanjiToJaxMARL
from envs.overcooked.overcooked_wrapper import OvercookedWrapper
from envs.overcooked.augmented_layouts import augmented_layouts

def process_default_args(env_kwargs: dict, default_args: dict):
    '''Helper function to process generator and viewer args for Jumanji environments. 
    If env_args and default_args have any key overlap, overwrite 
    args in default_args with those in env_args, deleting those in env_args
    '''
    env_kwargs_copy = dict(copy.deepcopy(env_kwargs))
    default_args_copy = dict(copy.deepcopy(default_args))
    for key in env_kwargs:
        if key in default_args:
            default_args_copy[key] = env_kwargs[key]
            del env_kwargs_copy[key]
    return default_args_copy, env_kwargs_copy

def make_env(env_name: str, env_kwargs: dict = {}):
    if env_name == "lbf":
        default_generator_args = {"grid_size": 7, "fov": 7, 
                          "num_agents": 2, "num_food": 3, 
                          "max_agent_level": 2, "force_coop": True}
        default_viewer_args = {"highlight_agent_idx": 0} # None to disable highlighting

        generator_args, env_kwargs_copy = process_default_args(env_kwargs, default_generator_args)
        viewer_args, env_kwargs_copy = process_default_args(env_kwargs_copy, default_viewer_args)
        env = jumanji.make('LevelBasedForaging-v0', 
                            generator=LbfGenerator(**generator_args),
                            **env_kwargs_copy,
                            viewer=AdHocLBFViewer(grid_size=generator_args["grid_size"],
                                                  **viewer_args))
        env = JumanjiToJaxMARL(env)
        
    elif env_name == 'overcooked-v1':
        default_env_kwargs = {"random_reset": True, "random_obj_state": False, "max_steps": 400}
        env_kwargs_copy = dict(copy.deepcopy(env_kwargs))
        # add default args that are not already in env_kwargs
        for key in default_env_kwargs:
            if key not in env_kwargs:
                env_kwargs_copy[key] = default_env_kwargs[key]

        layout = augmented_layouts[env_kwargs['layout']]
        env_kwargs_copy["layout"] = layout
        env = OvercookedWrapper(**env_kwargs_copy)
    else:
        env = jaxmarl.make(env_name, **env_kwargs)
    return env
