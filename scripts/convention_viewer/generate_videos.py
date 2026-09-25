"""Render videos of each evaluation teammate playing with its best response (BR).

For a task (LBF layout or Overcooked layout), loads every entry of
`evaluation/configs/global_heldout_settings.yaml:heldout_set.<task>`, pairs it
with the matching BR from `evaluation/configs/global_heldout_br.yaml`, rolls out
episodes, and writes an mp4 per teammate plus a `manifest.json` that drives the
convention viewer site. Teammates without a BR fall back to self-play.

Requires a GPU for orbax deserialization. From the repo root:

  PYTHONPATH=. python scripts/convention_viewer/generate_videos.py --task coord_ring
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs import make_env
from evaluation.heldout_core import load_heldout_set
from scripts.convention_viewer.tasks import TASKS, TaskSpec, sanitize

log = logging.getLogger("convention_viewer.videos")

HELDOUT_CFG = REPO_ROOT / "evaluation" / "configs" / "global_heldout_settings.yaml"
BR_CFG = REPO_ROOT / "evaluation" / "configs" / "global_heldout_br.yaml"


def build_env(task: TaskSpec):
    """Env + env_kwargs matching the population-diversity rollout settings."""
    if task.env == "lbf":
        from scripts.population_diversity.compute_population_diversity import (
            LBF_VARIANTS,
        )

        cfg = LBF_VARIANTS[task.variant]
        env_kwargs = {
            "grid_size": cfg["grid_size"],
            "num_food": cfg["num_food"],
            "different_levels": cfg["different_levels"],
        }
    else:
        env_kwargs = {
            "layout": task.variant,
            "random_obj_state": True,
            "do_reward_shaping": True,
            "reward_shaping_params": {
                "PLACEMENT_IN_POT_REW": 0.5,
                "PLATE_PICKUP_REWARD": 0.1,
                "SOUP_PICKUP_REWARD": 1.0,
                "ONION_PICKUP_REWARD": 0.1,
                "COUNTER_PICKUP_REWARD": 0.0,
                "COUNTER_DROP_REWARD": 0.0,
            },
        }
    # NOTE: no LogWrapper here -- the visualizers expect the raw env states.
    return make_env(task.env_name, env_kwargs), env_kwargs


def _existing_paths_only(block: dict[str, Any]) -> tuple[dict[str, Any], list]:
    """Drop config entries whose checkpoints aren't present locally."""
    from common.save_load_utils import REPO_PATH

    kept, skipped = {}, []
    for name, cfg in (block or {}).items():
        if cfg is None:
            continue
        ok = True
        for key in ("path", "weight_file"):
            if key in cfg:
                p = cfg[key]
                resolved = p if os.path.isabs(p) else os.path.join(REPO_PATH, p)
                if not os.path.exists(resolved):
                    ok = False
                    skipped.append(f"{name} (missing {key}={p})")
                    break
        if ok:
            kept[name] = cfg
    return kept, skipped


def load_teammates(task: TaskSpec, env, env_kwargs, seed: int) -> dict[str, tuple]:
    with HELDOUT_CFG.open() as fh:
        heldout_set = yaml.safe_load(fh)["heldout_set"]
    if task.yaml_key not in heldout_set:
        raise KeyError(f"heldout_set.{task.yaml_key} not in {HELDOUT_CFG}")
    block, skipped = _existing_paths_only(heldout_set[task.yaml_key])
    if skipped:
        log.warning(
            "skipping %d teammates without local checkpoints: %s",
            len(skipped),
            ", ".join(skipped),
        )
    return load_heldout_set(
        block, env, task.yaml_key, env_kwargs, jax.random.PRNGKey(seed)
    )


def load_brs(task: TaskSpec, env, env_kwargs, seed: int) -> dict[str, tuple]:
    """Map sanitized teammate key -> (policy, params, test_mode)."""
    with BR_CFG.open() as fh:
        br_set = yaml.safe_load(fh).get("best_response_set", {})
    if task.yaml_key not in br_set:
        log.warning(
            "no best_response_set block for %s; all teammates fall back to self-play",
            task.yaml_key,
        )
        return {}
    block, skipped = _existing_paths_only(br_set[task.yaml_key])
    if skipped:
        log.warning(
            "skipping %d BRs without local checkpoints: %s",
            len(skipped),
            ", ".join(skipped),
        )
    if not block:
        return {}
    loaded = load_heldout_set(
        block, env, task.yaml_key, env_kwargs, jax.random.PRNGKey(seed + 1)
    )
    out = {}
    for label, entry in loaded.items():
        base = label.split(" (")[0]  # 'br_for_x (0)' -> 'br_for_x'
        key = base.removeprefix("br_for_")
        out[sanitize(key)] = (entry[0], entry[1], entry[2])
    return out


def rollout_states(
    env,
    policy_0,
    params_0,
    test_mode_0,
    policy_1,
    params_1,
    test_mode_1,
    max_steps: int,
    rng,
):
    """Run one episode, returning the list of env states and the team return."""
    rng, reset_rng = jax.random.split(rng)
    obs, env_state = env.reset(reset_rng)
    done = {k: jnp.zeros((1), dtype=bool) for k in env.agents + ["__all__"]}
    hstate_0 = policy_0.init_hstate(1, aux_info={"agent_id": 0})
    hstate_1 = policy_1.init_hstate(1, aux_info={"agent_id": 1})

    states = [env_state]
    ep_return = 0.0
    for _ in range(max_steps):
        avail = jax.lax.stop_gradient(env.get_avail_actions(env_state))
        rng, a0_rng, a1_rng, step_rng = jax.random.split(rng, 4)
        act_0, hstate_0 = policy_0.get_action(
            params=params_0,
            obs=obs["agent_0"].reshape(1, 1, -1),
            done=done["agent_0"].reshape(1, 1),
            avail_actions=avail["agent_0"].astype(jnp.float32),
            hstate=hstate_0,
            rng=a0_rng,
            aux_obs=None,
            env_state=env_state,
            test_mode=test_mode_0,
        )
        act_1, hstate_1 = policy_1.get_action(
            params=params_1,
            obs=obs["agent_1"].reshape(1, 1, -1),
            done=done["agent_1"].reshape(1, 1),
            avail_actions=avail["agent_1"].astype(jnp.float32),
            hstate=hstate_1,
            rng=a1_rng,
            aux_obs=None,
            env_state=env_state,
            test_mode=test_mode_1,
        )
        env_act = {"agent_0": int(act_0.squeeze()), "agent_1": int(act_1.squeeze())}
        obs, env_state, reward, done, _ = env.step(step_rng, env_state, env_act)
        ep_return += float(reward["agent_0"])
        states.append(env_state)
        if bool(done["__all__"]):
            break
    return states, ep_return


def render(task: TaskSpec, env, states, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if task.env == "lbf":
        anim = env.animate(states, interval=150)
        anim.save(str(out_path), writer="ffmpeg")
    else:
        from envs.overcooked.adhoc_overcooked_visualizer import (
            AdHocOvercookedVisualizer,
        )

        AdHocOvercookedVisualizer().animate_mp4(
            [s.env_state for s in states],
            env.agent_view_size,
            highlight_agent_idx=0,
            filename=str(out_path),
            pixels_per_tile=32,
            fps=25,
        )


def update_manifest(manifest_path: Path, task: TaskSpec, entry: dict):
    manifest = {"tasks": {}}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    block = manifest["tasks"].setdefault(
        task.slug, {"label": task.label, "teammates": []}
    )
    block["label"] = task.label
    block["env"] = task.env_name
    tms = block["teammates"]
    tms[:] = [t for t in tms if t["key"] != entry["key"]]
    tms.append(entry)
    tms.sort(key=lambda t: t["key"])
    manifest_path.write_text(json.dumps(manifest, indent=1))


def generate_for_task(
    task: TaskSpec,
    out_dir: Path,
    num_eps: int,
    seed: int,
    only: set | None,
    overwrite: bool,
):
    env, env_kwargs = build_env(task)
    log.info("loading teammates for %s", task.yaml_key)
    teammates = load_teammates(task, env, env_kwargs, seed)
    brs = load_brs(task, env, env_kwargs, seed)
    log.info("%d teammates, %d BRs matched", len(teammates), len(brs))

    video_dir = out_dir / "videos" / task.slug
    manifest_path = out_dir / "manifest.json"

    for label, entry in teammates.items():
        policy, params, test_mode = entry[0], entry[1], entry[2]
        key = sanitize(label)
        if only and key not in only and label not in only:
            continue
        out_path = video_dir / f"{key}.mp4"
        br = brs.get(key)
        partner = "BR" if br is not None else "self-play"
        if out_path.exists() and not overwrite:
            log.info("skip existing %s/%s", task.slug, key)
            continue
        if br is None:
            log.warning("no BR for %s; rolling out self-play instead", label)
            br_policy, br_params, br_test = policy, params, test_mode
        else:
            br_policy, br_params, br_test = br

        rng = jax.random.PRNGKey(112358 + seed)
        states, returns = [], []
        for ep in range(num_eps):
            rng, ep_rng = jax.random.split(rng)
            ep_states, ep_return = rollout_states(
                env,
                policy,
                params,
                test_mode,
                br_policy,
                br_params,
                br_test,
                task.max_steps,
                ep_rng,
            )
            states.extend(ep_states)
            returns.append(ep_return)
        render(task, env, states, out_path)
        mean_return = sum(returns) / len(returns)
        update_manifest(
            manifest_path,
            task,
            {
                "key": key,
                "name": label,
                "partner": partner,
                "num_episodes": num_eps,
                "mean_return": round(mean_return, 3),
                "file": f"videos/{task.slug}/{key}.mp4",
            },
        )
        log.info(
            "done %s/%s (partner=%s, mean return %.2f)",
            task.slug,
            key,
            partner,
            mean_return,
        )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--task",
        default="all",
        choices=["all", *TASKS],
        help="task slug to render (default: all)",
    )
    ap.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parent / "site"),
        help="site directory that holds videos/ and manifest.json",
    )
    ap.add_argument(
        "--num-eps",
        type=int,
        default=None,
        help="episodes per teammate (default: 2 for LBF, 1 for Overcooked)",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--agents",
        default=None,
        help="comma-separated teammate keys to render (default: all)",
    )
    ap.add_argument(
        "--overwrite", action="store_true", help="re-render videos that already exist"
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True
    )
    out_dir = Path(args.out_dir)
    only = {s.strip() for s in args.agents.split(",")} if args.agents else None
    slugs = list(TASKS) if args.task == "all" else [args.task]

    for slug in slugs:
        task = TASKS[slug]
        num_eps = (
            args.num_eps
            if args.num_eps is not None
            else (2 if task.env == "lbf" else 1)
        )
        generate_for_task(task, out_dir, num_eps, args.seed, only, args.overwrite)
    return 0


if __name__ == "__main__":
    sys.exit(main())
