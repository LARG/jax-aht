import copy

import jaxmarl
import jumanji
from jumanji.environments.routing.lbf.generator import RandomGenerator as LbfGenerator

from envs.lbf.adhoc_lbf_viewer import AdHocLBFViewer
from envs.lbf.lbf_wrapper import LBFWrapper
from envs.lbf.reward_shaping_lbf_wrapper import RewardShapingLBFWrapper
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
    if env_name in ['lbf', 'lbf-reward-shaping']:
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

        if env_name == 'lbf-reward-shaping':
            env = RewardShapingLBFWrapper(env, share_rewards=True)
        else:
            env = LBFWrapper(env, share_rewards=True)
        
    elif env_name == 'overcooked-v1':
        default_env_kwargs = {"random_reset": True, "random_obj_state": False, "max_steps": 400}
        
        # preprocess env_kwargs to maintain compatibility with symmetric reward shaping
        if "reward_shaping_params" in env_kwargs:
            for param in env_kwargs["reward_shaping_params"]:
                payload = env_kwargs["reward_shaping_params"][param]
                if type(payload) == int or type(payload) == float:
                    # turn the param into symmetric form
                    env_kwargs["reward_shaping_params"][param] = [payload, payload] 
                elif type(payload) == tuple or type(payload) == list:
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

        layout = augmented_layouts[env_kwargs['layout']]
        env_kwargs_copy["layout"] = layout
        env = OvercookedWrapper(**env_kwargs_copy)

    elif env_name == 'coin-game':
        from envs.coins.coins import CoinGameWrapper
        default_env_kwargs = {"num_agents": 2, "grid_size": 7, "max_steps": 1000}
        env_kwargs_copy = dict(copy.deepcopy(env_kwargs))
        for key in default_env_kwargs:
            if key not in env_kwargs:
                env_kwargs_copy[key] = default_env_kwargs[key]
        env = CoinGameWrapper(**env_kwargs_copy)

    else:
        env = jaxmarl.make(env_name, **env_kwargs)
    return env
