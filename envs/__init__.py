import copy

import jaxmarl
import jumanji
from jumanji.environments.routing.lbf.generator import RandomGenerator as LbfGenerator
from jumanji.environments.routing.robot_warehouse.generator import RandomGenerator as RWAREGenerator
from envs.jumanji_jaxmarl_wrapper import JumanjiToJaxMARL
from envs.overcooked.overcooked_wrapper import OvercookedWrapper
from envs.overcooked.augmented_layouts import augmented_layouts


def process_generator_args(env_kwargs: dict, default_generator_args: dict):
    '''if env_args and default_generator_args have any key overlap, overwrite 
    args in default_generator_args with those in env_args, deleting those in env_args
    '''
    env_kwargs_copy = dict(copy.deepcopy(env_kwargs))
    generator_args_copy = dict(copy.deepcopy(default_generator_args))
    for key in env_kwargs:
        if key in default_generator_args:
            generator_args_copy[key] = env_kwargs[key]
            del env_kwargs_copy[key]
    return generator_args_copy, env_kwargs_copy

def make_env(env_name: str, env_kwargs: dict = {}):
    if env_name == "lbf":
        default_generator_args = {"grid_size": 7, "fov": 7, 
                          "num_agents": 2, "num_food": 3, 
                          "max_agent_level": 2, "force_coop": True}
        generator_args, env_kwargs_copy = process_generator_args(env_kwargs, default_generator_args)
        env = jumanji.make('LevelBasedForaging-v0', 
                            generator=LbfGenerator(**generator_args),
                            **env_kwargs_copy)
        env = JumanjiToJaxMARL(env)

    elif env_name == "robot-warehouse-2p":
        default_generator_args = {"shelf_rows": 2, "shelf_columns": 3, 
                          "column_height": 8, "num_agents": 2, 
                          "sensor_range": 1, "request_queue_size": 8}
        generator_args, env_kwargs_copy = process_generator_args(env_kwargs, default_generator_args)
        env = jumanji.make("RobotWarehouse-v0", 
                            generator=RWAREGenerator(**generator_args),
                            **env_kwargs_copy)
        env = JumanjiToJaxMARL(env)

    elif env_name == "overcooked-v2":
        layout = augmented_layouts[env_kwargs["layout"]]
        env_kwargs_copy = copy.deepcopy(env_kwargs)
        env_kwargs_copy["layout"] = layout
        env = OvercookedWrapper(**env_kwargs_copy)
    else:
        env = jaxmarl.make(env_name, **env_kwargs)
    return env
