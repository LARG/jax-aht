import os

base_dir = "/Users/jeffrey/Desktop/AHTMD/jax-aht/social_laws/configs/task/continuous"

def create_task_yaml(n, law_active, single_task):
    if law_active:
        prefix = "law"
        social_min_dist = 0.20
        min_sep_goal = 0.30
    else:
        prefix = "no_law"
        social_min_dist = 0.0
        min_sep_goal = 0.0

    if single_task:
        filename = f"coop_recon_compare_{prefix}_{n}_agent_single_task.yaml"
        task_name = f"CoopRecon_Compare_{prefix.capitalize()}_{n}AgentProjection"
        env_name = "continuous/coop_recon_sap"
    else:
        filename = f"coop_recon_compare_{prefix}_{n}_agent.yaml"
        task_name = f"CoopRecon_Compare_{prefix.capitalize()}_{n}Agent"
        env_name = "continuous/coop_recon_n_agent"

    content = f"""ENV_NAME: {env_name}
TASK_NAME: {task_name}
ROLLOUT_LENGTH: 128
NUM_EVAL_EPISODES: 50

"""
    if not single_task:
        for i in range(1, n+1):
            content += f"SINGLE_AGENT_{i}_PROJECTION: continuous/coop_recon_compare_{prefix}_{n}_agent_single_task\n"
        content += "\n"

    content += f"""ENV_KWARGS:
  num_agents: {n}
  grid_size: 1.5
  ego_centric_obs: true
  movement_noise_std: 0.1
  done_condition: "all"
  horizon: 150
  render: false
  render_name: "{task_name.lower()}"
  social_min_dist: {social_min_dist}
  min_sep_goal: {min_sep_goal}
  collision_radius: 0.05

ALPHA_COST: false
"""
    
    filepath = os.path.join(base_dir, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write(content)
        
for n in [2, 3, 4, 5]:
    for law in [True, False]:
        for single in [True, False]:
            create_task_yaml(n, law, single)
print("done")
