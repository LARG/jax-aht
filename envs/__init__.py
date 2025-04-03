import copy

import jaxmarl
import jumanji
from jumanji.environments.routing.lbf.generator import RandomGenerator

from envs.jumanji_jaxmarl_wrapper import JumanjiToJaxMARL
from envs.overcooked.overcooked_wrapper import OvercookedWrapper
from envs.overcooked.augmented_layouts import augmented_layouts

def make_env(env_name: str, env_kwargs: dict = {}):
    if env_name == 'lbf':
        generator_args = {"grid_size": 8, "fov": 8, 
                          "num_agents": 2, "num_food": 3, 
                          "max_agent_level": 2, "force_coop": False}
        # if env_args and default_generator_args have any key overlap, replace 
        # args in default_generator_args with those in env_args, deleting those in env_args
        env_kwargs_copy = copy.deepcopy(env_kwargs)
        for key in env_kwargs_copy:
            if key in generator_args:
                generator_args[key] = env_kwargs[key]
                del env_kwargs[key]
        env = jumanji.make('LevelBasedForaging-v0', 
                            generator=RandomGenerator(**generator_args),
                            **env_kwargs)
        env = JumanjiToJaxMARL(env)
        
    elif env_name == 'overcooked-v2':
        layout = augmented_layouts[env_kwargs['layout']]
        env_kwargs_copy = copy.deepcopy(env_kwargs)
        env_kwargs_copy["layout"] = layout
        env = OvercookedWrapper(**env_kwargs_copy)
    else:
        env = jaxmarl.make(env_name, **env_kwargs)
    return env
