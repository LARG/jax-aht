import copy
import numpy as np

from omegaconf import ListConfig

import jaxmarl
import jumanji
from jumanji.environments.routing.lbf.generator import RandomGenerator as LbfGenerator

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
    if env_name in ['lbf', 'lbf-reward-shaping']:
        # LBF options can be passed as kwargs with sensible defaults
        #   grid_size (int): default 7
        #   num_food (int): default 3
        #   different_levels (bool): default False

        # Standard configs used in human_data collection
        # 1. grid_size=7, num_food=3, different_levels=False (default)
        # 2. grid_size=7, num_food=3, different_levels=True
        # 3. grid_size=12, num_food=6, different_levels=False
        # 4. grid_size=12, num_food=6, different_levels=True

        # Ex: task.ENV_KWARGS.grid_size=12 task.ENV_KWARGS.num_food=6

        from envs.lbf.adhoc_lbf_viewer import AdHocLBFViewer
        from envs.lbf.different_levels_generator import DifferentLevelsGenerator
        from envs.lbf.lbf_wrapper import LBFWrapper
        from envs.lbf.reward_shaping_lbf_wrapper import RewardShapingLBFWrapper

        grid_size = env_kwargs.get("grid_size", 7)
        num_food = env_kwargs.get("num_food", 3)
        different_levels = env_kwargs.get("different_levels", False)

        default_generator_args = {
            "grid_size": grid_size,
            "fov": grid_size, 
            "num_agents": 2,
            "num_food": num_food, 
            "max_agent_level": 2,
            "force_coop": True,
        }
        default_viewer_args = {"highlight_agent_idx": 0} # None to disable highlighting

        ignore_keys = ["different_levels"]
        generator_args, env_kwargs_copy = process_default_args(env_kwargs, default_generator_args)
        viewer_args, env_kwargs_copy = process_default_args(env_kwargs_copy, default_viewer_args)

        # remove "different_levels" from env_kwargs_copy since it's not an argument for the environment
        if "different_levels" in env_kwargs_copy:
            del env_kwargs_copy["different_levels"]
        
        generator = DifferentLevelsGenerator(**generator_args) if different_levels else LbfGenerator(**generator_args)

        env = jumanji.make('LevelBasedForaging-v0', 
                            generator=generator,
                            **env_kwargs_copy,
                            viewer=AdHocLBFViewer(grid_size=generator_args["grid_size"],
                                                  **viewer_args))

        if env_name == 'lbf-reward-shaping':
            env = RewardShapingLBFWrapper(env, share_rewards=True)
        else:
            env = LBFWrapper(env, share_rewards=True)

    elif env_name == 'overcooked-v1':
        default_env_kwargs = {
            "random_reset": True,
            "random_obj_state": False,
            "max_steps": 400
        }

        # preprocess env_kwargs to maintain compatibility with symmetric reward shaping
        if "reward_shaping_params" in env_kwargs:
            for param in env_kwargs["reward_shaping_params"]:
                payload = env_kwargs["reward_shaping_params"][param]
                if type(payload) == int or type(payload) == float:
                    # turn the param into symmetric form
                    env_kwargs["reward_shaping_params"][param] = [payload, payload]
                elif type(payload) == tuple or type(payload) == list or type(payload) == ListConfig:
                    # this is the correct format
                    pass
                else:
                    print(f"\n[Environment Instantiation Error] {type(payload)} is not valid type as a reward shaping parameter for {param}.\n")
                    exit()

        env_kwargs_copy = dict(copy.deepcopy(env_kwargs))
        # add default args that are not already in env_kwargs
        for key in default_env_kwargs:
            if key not in env_kwargs:
                env_kwargs_copy[key] = default_env_kwargs[key]

        from envs.overcooked_v1.augmented_layouts import augmented_layouts
        from envs.overcooked_v1.overcooked_wrapper import OvercookedWrapper

        layout = augmented_layouts[env_kwargs['layout']]
        env_kwargs_copy["layout"] = layout
        env = OvercookedWrapper(**env_kwargs_copy)

    elif env_name == 'hanabi':
        default_env_kwargs = {
            "num_agents": 2,
            "num_colors": 5,
            "num_ranks": 5,
            "max_info_tokens": 8,
            "max_life_tokens": 3,
            "num_cards_of_rank": np.array([3, 2, 2, 2, 1]),
        }

        from envs.hanabi.hanabi_wrapper import HanabiWrapper
        env_kwargs = default_env_kwargs
        env = HanabiWrapper(**env_kwargs)

    else:
        raise NotImplementedError(f"Environment {env_name} not implemented in make_env.")

    return env

if __name__ == "__main__":
    # sanity check: test environment creation
    env = make_env('lbf-reward-shaping', {'num_agents': 3, 'grid_size': 9})
    print(env)
    env = make_env('overcooked-v1', {'layout': 'cramped_room'})
    print(env)
    env = make_env('hanabi', {'num_agents': 2})
    print(env)