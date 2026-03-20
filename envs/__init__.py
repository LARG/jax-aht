import copy
import numpy as np
import os
from random import seed

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
        default_generator_args = {
            "grid_size": 7,
            "fov": 7,
            "num_agents": 2,
            "num_food": 3,
            "max_agent_level": 2,
            "force_coop": True,
        }
        default_viewer_args = {"highlight_agent_idx": 0} # None to disable highlighting

        from envs.lbf.lbf_wrapper import LBFWrapper
        from envs.lbf.reward_shaping_lbf_wrapper import RewardShapingLBFWrapper
        from envs.lbf.adhoc_lbf_viewer import AdHocLBFViewer

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

        from envs.overcooked.augmented_layouts import augmented_layouts
        from envs.overcooked.overcooked_wrapper import OvercookedWrapper

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

    elif env_name == 'rddl/grid_4x4':
        default_env_kwargs = {
            "domain": "grid_4x4_domain.rddl",
            "instance": "grid_4x4_instance2.rddl",
            "render": False,
            "render_name": "grid_4x4",
            "render_dir": "render",
            "stochastic_movement_prob": 0.0,
            "enforce_action_constraints": True,
            "vectorized": True,
            "ego_centric_obs": False
        }

        from pyRDDLGym_jax.core.env import JaxRDDLEnv
        from envs.rddl.grid_4x4.grid_4x4_wrapper import Grid4x4Wrapper
        env_kwargs_copy = dict(copy.deepcopy(env_kwargs))
        # add default args that are not already in env_kwargs
        for key in default_env_kwargs:
            if key not in env_kwargs:
                env_kwargs_copy[key] = default_env_kwargs[key]

        # create the JAX RDDL Grid4x4 environment
        rddl_env = JaxRDDLEnv(domain=os.path.join(os.path.dirname(__file__), 'rddl/grid_4x4', env_kwargs_copy["domain"]),
                              instance=os.path.join(os.path.dirname(__file__), 'rddl/grid_4x4', env_kwargs_copy["instance"]),
                              vectorized=env_kwargs_copy["vectorized"], randomizable_nonfluents=["GOAL", "OBSTACLE"])
        env = Grid4x4Wrapper(rddl_env, **env_kwargs_copy)

    elif env_name == 'rddl/grid_4x4_alternating':
        default_env_kwargs = {
            "domain": "grid_4x4_alternating_domain.rddl",
            "instance": "grid_4x4_alternating_instance2.rddl",
            "render": False,
            "render_name": "grid_4x4_alternating",
            "render_dir": "render",
            "stochastic_movement_prob": 0.0,
            "enforce_action_constraints": True,
            "vectorized": True,
            "ego_centric_obs": False
        }

        from pyRDDLGym_jax.core.env import JaxRDDLEnv
        from envs.rddl.grid_4x4_alternating.grid_4x4_alternating_wrapper import Grid4x4AlternatingWrapper
        env_kwargs_copy = dict(copy.deepcopy(env_kwargs))
        # add default args that are not already in env_kwargs
        for key in default_env_kwargs:
            if key not in env_kwargs:
                env_kwargs_copy[key] = default_env_kwargs[key]

        # create the JAX RDDL Grid4x4 environment
        rddl_env = JaxRDDLEnv(domain=os.path.join(os.path.dirname(__file__), 'rddl/grid_4x4_alternating', env_kwargs_copy["domain"]),
                              instance=os.path.join(os.path.dirname(__file__), 'rddl/grid_4x4_alternating', env_kwargs_copy["instance"]),
                              vectorized=env_kwargs_copy["vectorized"], randomizable_nonfluents=["GOAL", "OBSTACLE"])
        env = Grid4x4AlternatingWrapper(rddl_env, **env_kwargs_copy)

    elif env_name == 'rddl/grid_10x10':
        default_env_kwargs = {
            "domain": "grid_10x10_domain.rddl",
            "instance": "grid_10x10_instance2.rddl",
            "render": False,
            "render_name": "grid_10x10",
            "render_dir": "render",
            "stochastic_movement_prob": 0.0,
            "enforce_action_constraints": True,
            "vectorized": True,
            "ego_centric_obs": False
        }

        from pyRDDLGym_jax.core.env import JaxRDDLEnv
        from envs.rddl.grid_10x10.grid_10x10_wrapper import Grid10x10Wrapper
        env_kwargs_copy = dict(copy.deepcopy(env_kwargs))
        # add default args that are not already in env_kwargs
        for key in default_env_kwargs:
            if key not in env_kwargs:
                env_kwargs_copy[key] = default_env_kwargs[key]

        # create the JAX RDDL Grid10x10 environment
        rddl_env = JaxRDDLEnv(domain=os.path.join(os.path.dirname(__file__), 'rddl/grid_10x10', env_kwargs_copy["domain"]),
                              instance=os.path.join(os.path.dirname(__file__), 'rddl/grid_10x10', env_kwargs_copy["instance"]),
                              vectorized=env_kwargs_copy["vectorized"], randomizable_nonfluents=["GOAL", "OBSTACLE"])
        env = Grid10x10Wrapper(rddl_env, **env_kwargs_copy)

    elif env_name == 'rddl/grid_10x10_alternating':
        default_env_kwargs = {
            "domain": "grid_10x10_alternating_domain.rddl",
            "instance": "grid_10x10_alternating_instance2.rddl",
            "render": False,
            "render_name": "grid_10x10_alternating",
            "render_dir": "render",
            "stochastic_movement_prob": 0.0,
            "enforce_action_constraints": True,
            "vectorized": True,
            "ego_centric_obs": False,
            "single_task": False
        }

        from pyRDDLGym_jax.core.env import JaxRDDLEnv
        from envs.rddl.grid_10x10_alternating.grid_10x10_alternating_wrapper import Grid10x10AlternatingWrapper
        env_kwargs_copy = dict(copy.deepcopy(env_kwargs))
        # add default args that are not already in env_kwargs
        for key in default_env_kwargs:
            if key not in env_kwargs:
                env_kwargs_copy[key] = default_env_kwargs[key]

        # create the JAX RDDL Grid10x10 environment
        rddl_env = JaxRDDLEnv(domain=os.path.join(os.path.dirname(__file__), 'rddl/grid_10x10_alternating', env_kwargs_copy["domain"]),
                              instance=os.path.join(os.path.dirname(__file__), 'rddl/grid_10x10_alternating', env_kwargs_copy["instance"]),
                              vectorized=env_kwargs_copy["vectorized"], randomizable_nonfluents=["GOAL", "OBSTACLE"])
        env = Grid10x10AlternatingWrapper(rddl_env, **env_kwargs_copy)

    elif env_name == 'rddl/pizza':
        default_env_kwargs = {
            "domain": "pizza_domain_new.rddl",
            "instance": "pizza_instance_all.rddl",
            "render": False,
            "render_name": "pizza",
            "render_dir": "render",
            "enforce_action_constraints": True,
            "vectorized": True,
            "ego_centric_obs": False
        }

        from pyRDDLGym_jax.core.env import JaxRDDLEnv
        from envs.rddl.pizza.pizza_wrapper import PizzaWrapper
        env_kwargs_copy = dict(copy.deepcopy(env_kwargs))
        # add default args that are not already in env_kwargs
        for key in default_env_kwargs:
            if key not in env_kwargs:
                env_kwargs_copy[key] = default_env_kwargs[key]

        # create the JAX RDDL Pizza environment
        rddl_env = JaxRDDLEnv(domain=os.path.join(os.path.dirname(__file__), 'rddl/pizza', env_kwargs_copy["domain"]),
                              instance=os.path.join(os.path.dirname(__file__), 'rddl/pizza', env_kwargs_copy["instance"]),
                              vectorized=env_kwargs_copy["vectorized"])
        env = PizzaWrapper(rddl_env, **env_kwargs_copy)

    elif env_name == 'rddl/pizza_v2':
        default_env_kwargs = {
            "domain": "pizza_v2_domain.rddl",
            "instance": "pizza_v2_instance_all.rddl",
            "render": False,
            "render_name": "pizza_v2",
            "render_dir": "render",
            "enforce_action_constraints": True,
            "vectorized": True,
            "ego_centric_obs": False
        }

        from pyRDDLGym_jax.core.env import JaxRDDLEnv
        from envs.rddl.pizza_v2.pizza_v2_wrapper import PizzaWrapper
        env_kwargs_copy = dict(copy.deepcopy(env_kwargs))
        # add default args that are not already in env_kwargs
        for key in default_env_kwargs:
            if key not in env_kwargs:
                env_kwargs_copy[key] = default_env_kwargs[key]

        # create the JAX RDDL Pizza V2 environment
        rddl_env = JaxRDDLEnv(domain=os.path.join(os.path.dirname(__file__), 'rddl/pizza_v2', env_kwargs_copy["domain"]),
                              instance=os.path.join(os.path.dirname(__file__), 'rddl/pizza_v2', env_kwargs_copy["instance"]),
                              vectorized=env_kwargs_copy["vectorized"])
        env = PizzaWrapper(rddl_env, **env_kwargs_copy)

    elif env_name == 'continuous/coop_recon':
        default_env_kwargs = {
            "instance": "",
            "render": False,
            "render_name": "coop_recon_continuous",
            "render_dir": "render",
            "enforce_action_constraints": True,
            "ego_centric_obs": False
        }

        from envs.coop_recon_continuous.coop_recon_continuous_wrapper import CoopReconContinuousWrapper
        env_kwargs_copy = dict(copy.deepcopy(env_kwargs))
        # add default args that are not already in env_kwargs
        for key in default_env_kwargs:
            if key not in env_kwargs:
                env_kwargs_copy[key] = default_env_kwargs[key]

        # create the CoopReconContinuous environment
        env = CoopReconContinuousWrapper(**env_kwargs_copy)

    elif env_name == 'continuous/coop_recon_n_agent':
        # Phase B: N-agent generalization (N=3 or N=4).
        # Use num_agents kwarg to set N; grid_size kwarg to scale the arena.
        default_env_kwargs = {
            "instance": "",
            "render": False,
            "render_name": "coop_recon_n_agent",
            "render_dir": "render",
            "ego_centric_obs": False,
            "num_agents": 3,
            "grid_size": 1.0,
        }

        from envs.coop_recon_continuous.coop_recon_continuous_n_agent_wrapper import CoopReconContinuousNAgentWrapper
        env_kwargs_copy = dict(copy.deepcopy(env_kwargs))
        for key in default_env_kwargs:
            if key not in env_kwargs:
                env_kwargs_copy[key] = default_env_kwargs[key]

        env = CoopReconContinuousNAgentWrapper(**env_kwargs_copy)

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
