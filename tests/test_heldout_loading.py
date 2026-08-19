"""Smoke test: load every heldout agent for every task in global_heldout_settings.yaml.

Catches config/checkpoint mismatches (bad paths, missing idx_list, wrong
actor_type routing) without running any evaluation rollouts.

Run with: PYTHONPATH=. python tests/test_heldout_loading.py
Requires the eval_teammates data (see download_eval_data.py).
"""
import os

import jax
import yaml

from envs import make_env
from envs.log_wrapper import LogWrapper
from evaluation.heldout_evaluator import load_heldout_set

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HELDOUT_CONFIG = os.path.join(REPO_ROOT, "evaluation/configs/global_heldout_settings.yaml")
TASK_CONFIG_DIR = os.path.join(REPO_ROOT, "evaluation/configs/task")


def load_task_config(task_name):
    with open(os.path.join(TASK_CONFIG_DIR, f"{task_name}.yaml")) as f:
        return yaml.safe_load(f)


def test_load_all_heldout_agents():
    with open(HELDOUT_CONFIG) as f:
        heldout_set = yaml.safe_load(f)["heldout_set"]

    rng = jax.random.PRNGKey(0)
    failures = []
    for task_name, heldout_config in heldout_set.items():
        task_cfg = load_task_config(task_name)
        env_kwargs = task_cfg["ENV_KWARGS"] or {}
        env = LogWrapper(make_env(task_cfg["ENV_NAME"], env_kwargs))
        try:
            agents = load_heldout_set(heldout_config, env, task_name, env_kwargs, rng)
        except FileNotFoundError as e:
            # Checkpoint data not downloaded on this machine (see download_eval_data.py).
            print(f"[skip] {task_name}: {e}")
            continue
        except Exception as e:
            failures.append((task_name, e))
            print(f"[FAIL] {task_name}: {type(e).__name__}: {e}")
            continue
        assert agents, f"{task_name}: no heldout agents loaded"
        print(f"[ok] {task_name}: loaded {len(agents)} agents: {sorted(agents)}")

    assert not failures, f"Failed to load heldout agents for: {[t for t, _ in failures]}"


if __name__ == "__main__":
    test_load_all_heldout_agents()
    print("All heldout sets loaded successfully.")
