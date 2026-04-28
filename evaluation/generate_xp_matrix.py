'''This script generates a XP matrix for the heldout set.

Run via `python evaluation/run.py -cn heldout_xp task=lbf`.
'''
import jax

from common.run_episodes import run_episodes
from common.tree_utils import tree_stack
from common.plot_utils import get_metric_names
from envs import make_env
from envs.log_wrapper import LogWrapper
from evaluation.heldout_evaluator import load_heldout_set, print_metrics_table, normalize_metrics

def heldout_crossplay(config, env, rng, num_episodes, heldout_agents, br_agents):
    '''Evaluate all heldout agents against each other
    Args: 
        heldout_agents: a dict of (policy, params, test_mode) tuples for each heldout partner. params might be None for heuristic agents.
        br_agents: dict with same structure as heldout_agents, but for best response agents.
    Returns a pytree of shape (num_heldout_agents, num_heldout_agents, num_eval_episodes, num_agents_per_env)
    '''
    num_agents = env.num_agents
    assert num_agents == 2, "This eval code assumes exactly 2 agents."

    heldout_agent_names = list(heldout_agents.keys())
    heldout_agent_list = list(heldout_agents.values())
    br_agent_names = list(br_agents.keys())
    br_agent_list = list(br_agents.values())
    
    num_heldout_agents = len(heldout_agent_list)

    def eval_pair_fn(rng, policy1, param1, test_mode1, policy2, param2, test_mode2):
        return run_episodes(rng, env, 
                            agent_0_param=param1, agent_0_policy=policy1,
                            agent_1_param=param2, agent_1_policy=policy2,
                            max_episode_steps=config["global_heldout_settings"]["MAX_EPISODE_STEPS"],
                            num_eps=num_episodes, 
                            agent_0_test_mode=test_mode1,
                            agent_1_test_mode=test_mode2)

    # Initialize results array
    all_metrics = []
    
    # Split RNG for each heldout agent
    rng, sub_rng = jax.random.split(rng)
    outer_heldout_rngs = jax.random.split(sub_rng, num_heldout_agents)
    
    # Double for loop implementation is necessary because the heldout agents have heterogeneous policy and 
    # param structures. 
    warned_missing_row_bounds = set()

    for i in range(num_heldout_agents):
        heldout_agent1 = heldout_agent_list[i]
        policy1, param1, test_mode1, performance_bounds1 = heldout_agent1
        rng1 = outer_heldout_rngs[i]
        
        # Split RNG for each heldout partner
        rng1, sub_rng1 = jax.random.split(rng1)
        partner_rngs = jax.random.split(sub_rng1, num_heldout_agents)
        
        partner_i_metrics = []
        for j in range(num_heldout_agents):
            br_agent = br_agent_list[j]
            policy2, param2, test_mode2, performance_bounds2 = br_agent
            rng2 = partner_rngs[j]
            
            # Evaluate the pair
            eval_metrics = eval_pair_fn(rng2, policy1, param1, test_mode1, policy2, param2, test_mode2)

            if config["global_heldout_settings"]["NORMALIZE_RETURNS"]:
                # XP row normalization: use the heldout-row agent bounds only.
                if performance_bounds1 is not None:
                    eval_metrics = normalize_metrics(eval_metrics, performance_bounds1)
                elif heldout_agent_names[i] not in warned_missing_row_bounds:
                    print(
                        f"Warning: missing row performance bounds for {heldout_agent_names[i]}; "
                        "skipping normalization for this entire row."
                    )
                    warned_missing_row_bounds.add(heldout_agent_names[i])

            partner_i_metrics.append(eval_metrics)

        all_metrics.append(tree_stack(partner_i_metrics))    
    return tree_stack(all_metrics)

def run_heldout_xp_evaluation(config, print_metrics=False):
    '''Run heldout evaluation'''
    # Create only one environment instance
    env = make_env(config["ENV_NAME"], config["ENV_KWARGS"])
    env = LogWrapper(env)
    metric_names = get_metric_names(config["ENV_NAME"])
    print("METRIC NAMES ARE: ", metric_names)

    rng = jax.random.PRNGKey(config["global_heldout_settings"]["EVAL_SEED"])
    rng, heldout_init_rng, br_init_rng, eval_rng = jax.random.split(rng, 4)
    
    # load heldout agents
    heldout_cfg = config["heldout_set"][config["TASK_NAME"]]
    heldout_agents = load_heldout_set(heldout_cfg, env, config["TASK_NAME"], config["ENV_KWARGS"], heldout_init_rng)
    
    # load best response agents
    br_cfg = config["best_response_set"][config["TASK_NAME"]]
    br_agents = load_heldout_set(br_cfg, env, config["TASK_NAME"], config["ENV_KWARGS"], br_init_rng)

    # sanity check
    assert len(heldout_agents) == len(br_agents), "Number of heldout agents and best response agents must be the same."
    # run evaluation
    eval_metrics = heldout_crossplay(
        config, env, eval_rng, config["global_heldout_settings"]["NUM_EVAL_EPISODES"], 
        heldout_agents, br_agents)

    if print_metrics:
        # each leaf of eval_metrics has shape (num_heldout_agents, num_heldout_agents, num_eval_episodes, num_agents_per_env)
        metric_names = get_metric_names(config["ENV_NAME"])
        heldout_names = list(heldout_agents.keys())
        br_names = list(br_agents.keys())
        save_heatmap = config.get("xp_matrix_outputs", {}).get("save_heatmap", False)
        for metric_name in metric_names:
            print_metrics_table(
                eval_metrics,
                metric_name,
                heldout_names,
                br_names,
                config["global_heldout_settings"]["AGGREGATE_STAT"],
                config["global_heldout_settings"]["NORMALIZE_RETURNS"],
                save=True,
                save_heatmap=save_heatmap,
            )
    return eval_metrics
