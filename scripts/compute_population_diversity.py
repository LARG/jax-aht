# Population Diversity CLI. Runs rollouts, computes PD, writes heatmap + PCA + features.csv + pd.json.
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs import make_env
from envs.log_wrapper import LogWrapper

from scripts.pd_events import (
    HANABI_FEATURE_NAMES,
    LBF_FEATURE_NAMES,
    OVERCOOKED_FEATURE_NAMES,
    hanabi_feature_names,
)
from scripts.pd_plots import (
    population_diversity,
    pca_2d,
    save_pca_plot,
    save_cosine_heatmap,
)
from scripts.pd_rollouts import (
    HanabiActionLayout,
    HanabiEpisodeCounts,
    hanabi_episode_to_vector,
    build_hanabi_agents,
    rollout_hanabi_two_policy,
    rollout_hanabi_self_play,
    LBFEpisodeCounts,
    lbf_episode_to_vector,
    lbf_step_update,
    build_lbf_agents,
    rollout_two_policy,
    rollout_simple_self_play,
    OvercookedEpisodeCounts,
    overcooked_episode_to_vector,
    overcooked_step_update,
    build_overcooked_agents,
    load_full_heldout_for_pd,
    load_brs_for_pd,
)

log = logging.getLogger("compute_pd")

@dataclass
class EnvConfig:
    name: str
    score_norm: float
    length_norm: float
    feature_names: List[str]
    build_env: Callable
    build_agents: Callable
    rollout: Callable                            # self-play rollout (legacy)
    to_vector: Callable[[Any, float, float], np.ndarray] = field(default=lambda *_: np.zeros(1))
    rollout_two_policy: Callable = field(default=lambda *args, **kwargs: [])  # BR-paired rollout


def hanabi_env_factory(
    num_colors: int,
    num_ranks: int,
    hand_size: int,
    card_counts: List[int],
    max_info_tokens: int = 8,
    max_life_tokens: int = 3,
):
    env_kwargs = {
        "num_agents": 2,
        "num_colors": num_colors,
        "num_ranks": num_ranks,
        "hand_size": hand_size,
        "max_info_tokens": max_info_tokens,
        "max_life_tokens": max_life_tokens,
        "num_cards_of_rank": np.array(card_counts),
    }
    env = make_env("hanabi", env_kwargs)
    env = LogWrapper(env)
    return env, env_kwargs


HANABI_VARIANTS = {
    "hanabi": dict(
        num_colors=5, num_ranks=5, hand_size=5,
        card_counts=[3, 2, 2, 2, 1],
        max_info_tokens=8, max_life_tokens=3,
        score_norm=25.0,
    ),
    "mini-hanabi": dict(
        num_colors=3, num_ranks=3, hand_size=3,
        card_counts=[2, 2, 1],
        max_info_tokens=5, max_life_tokens=3,
        score_norm=9.0,
    ),
}


def env_config_for_hanabi(variant: str) -> EnvConfig:
    if variant not in HANABI_VARIANTS:
        raise ValueError(f"unknown hanabi variant {variant!r}; expected one of {list(HANABI_VARIANTS)}")
    cfg = HANABI_VARIANTS[variant]

    def _build_env():
        return hanabi_env_factory(
            cfg["num_colors"], cfg["num_ranks"], cfg["hand_size"], cfg["card_counts"],
            max_info_tokens=cfg["max_info_tokens"],
            max_life_tokens=cfg["max_life_tokens"],
        )

    def _build_agents(env, env_kwargs):
        num_actions = env.action_space("agent_0").n
        layout = HanabiActionLayout(
            hand_size=cfg["hand_size"],
            num_colors=cfg["num_colors"],
            num_ranks=cfg["num_ranks"],
            num_actions=num_actions,
        )
        agents = build_hanabi_agents(
            hand_size=cfg["hand_size"],
            num_colors=cfg["num_colors"],
            num_ranks=cfg["num_ranks"],
            num_actions=num_actions,
            card_counts=np.array(cfg["card_counts"]),
        )
        return agents, layout

    def _rollout(env, policy, layout, num_eps, seed, params=None):
        return rollout_hanabi_self_play(env, policy, layout, num_eps, seed, params=params)

    def _rollout_two_policy(env, policy_a, params_a, policy_b, params_b, layout, num_eps, seed):
        return rollout_hanabi_two_policy(
            env, policy_a, params_a, policy_b, params_b, layout, num_eps, seed,
        )

    return EnvConfig(
        name=variant,
        score_norm=cfg["score_norm"],
        length_norm=200.0,
        feature_names=hanabi_feature_names(cfg["num_colors"], cfg["num_ranks"]),
        build_env=_build_env,
        build_agents=_build_agents,
        rollout=_rollout,
        rollout_two_policy=_rollout_two_policy,
        to_vector=hanabi_episode_to_vector,
    )



LBF_VARIANTS = {
    "lbf_7x7_nolevels": dict(grid_size=7, num_food=3, different_levels=False, num_fruits=3),
    "lbf_12x12": dict(grid_size=12, num_food=6, different_levels=True, num_fruits=6),
}


def env_config_for_lbf(variant: str = "lbf_7x7_nolevels") -> EnvConfig:
    from functools import partial
    if variant not in LBF_VARIANTS:
        raise ValueError(f"unknown lbf variant {variant!r}; expected one of {list(LBF_VARIANTS)}")
    cfg = LBF_VARIANTS[variant]

    def _build_env():
        env_kwargs = {"grid_size": cfg["grid_size"], "num_food": cfg["num_food"], "different_levels": cfg["different_levels"]}
        env = make_env("lbf", env_kwargs)
        env = LogWrapper(env)
        return env, env_kwargs

    def _build_agents(env, env_kwargs):
        agents = build_lbf_agents(grid_size=cfg["grid_size"], num_fruits=cfg["num_fruits"])
        return agents, None

    def _rollout(env, policy, layout, num_eps, seed, params=None):
        hz = min(int(getattr(getattr(env._env, "env", None), "time_limit", 128)), 128)
        return rollout_simple_self_play(
            env, policy, num_eps, seed, LBFEpisodeCounts, partial(lbf_step_update, horizon=hz), max_steps=128, params=params,
        )

    def _rollout_two_policy(env, policy_a, params_a, policy_b, params_b, layout, num_eps, seed):
        hz = min(int(getattr(getattr(env._env, "env", None), "time_limit", 128)), 128)
        return rollout_two_policy(
            env, policy_a, params_a, policy_b, params_b,
            num_eps, seed, LBFEpisodeCounts, partial(lbf_step_update, horizon=hz), max_steps=128,
        )

    return EnvConfig(
        name=f"lbf-{variant}",
        score_norm=1.0,
        length_norm=128.0,
        feature_names=LBF_FEATURE_NAMES,
        build_env=_build_env,
        build_agents=_build_agents,
        rollout=_rollout,
        rollout_two_policy=_rollout_two_policy,
        to_vector=lbf_episode_to_vector,
    )



def env_config_for_overcooked(layout_name: str = "cramped_room") -> EnvConfig:
    from envs.overcooked.augmented_layouts import augmented_layouts

    layout = augmented_layouts[layout_name]

    def _build_env():
        env_kwargs = {
            "layout": layout_name,
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
        env = make_env("overcooked-v1", env_kwargs)
        env = LogWrapper(env)
        return env, env_kwargs

    def _build_agents(env, env_kwargs):
        agents = build_overcooked_agents(layout=layout)
        return agents, None

    def _rollout(env, policy, layout_obj, num_eps, seed, params=None):
        return rollout_simple_self_play(
            env, policy, num_eps, seed, OvercookedEpisodeCounts, overcooked_step_update, max_steps=400, params=params,
        )

    def _rollout_two_policy(env, policy_a, params_a, policy_b, params_b, layout_obj, num_eps, seed):
        return rollout_two_policy(
            env, policy_a, params_a, policy_b, params_b,
            num_eps, seed, OvercookedEpisodeCounts, overcooked_step_update, max_steps=400,
        )

    return EnvConfig(
        name=f"overcooked-{layout_name}",
        score_norm=200.0,
        length_norm=400.0,
        feature_names=OVERCOOKED_FEATURE_NAMES,
        build_env=_build_env,
        build_agents=_build_agents,
        rollout=_rollout,
        rollout_two_policy=_rollout_two_policy,
        to_vector=overcooked_episode_to_vector,
    )



def compute_pd_for_env(
    env_cfg: EnvConfig,
    num_episodes: int,
    seed: int,
    output_dir: Path,
    full_heldout: bool = False,
    heldout_yaml_key: str | None = None,
    task_name: str | None = None,
    br_paired: bool = False,
    br_root: Path | None = None,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    env, env_kwargs = env_cfg.build_env()

    if full_heldout:
        if heldout_yaml_key is None or task_name is None:
            raise ValueError("full_heldout requires heldout_yaml_key + task_name")
        log.info("loading full heldout set from yaml key %r for task %r", heldout_yaml_key, task_name)
        heldout = load_full_heldout_for_pd(env, env_kwargs, task_name, heldout_yaml_key, seed)
        log.info("loaded %d agents from heldout_set.%s", len(heldout), heldout_yaml_key)
        agents = {label: (entry[0], entry[1]) for label, entry in heldout.items()}
        # Rebuild layout (Hanabi needs this; LBF/Overcooked layout=None)
        if env_cfg.name in HANABI_VARIANTS or env_cfg.name == "hanabi":
            cfg = HANABI_VARIANTS.get(env_cfg.name, HANABI_VARIANTS["hanabi"])
            num_actions = env.action_space("agent_0").n
            layout = HanabiActionLayout(
                hand_size=cfg["hand_size"],
                num_colors=cfg["num_colors"],
                num_ranks=cfg["num_ranks"],
                num_actions=num_actions,
            )
        else:
            layout = None
    else:
        agents_dict, layout = env_cfg.build_agents(env, env_kwargs)
        agents = {name: (policy, None) for name, policy in agents_dict.items()}

    # Load BRs if requested. Canonical ZSC-Eval PD pairs each partner with its
    # best-response. If a partner's BR is missing on disk, fall back to
    # self-play for THAT partner with a logged warning.
    brs: Dict[str, Tuple[Any, Any]] = {}
    if br_paired:
        if br_root is None:
            raise ValueError("br_paired requires br_root")
        if task_name is None:
            raise ValueError("br_paired requires task_name (e.g. 'mini-hanabi', 'lbf/lbf_7x7_nolevels')")
        # Derive layout_prefix from task_name for OC-style HF naming
        # (e.g. coord_ring_<name>_<idx>_serious). Strip 'overcooked-v1/' prefix.
        layout_prefix = None
        if task_name.startswith("overcooked-v1/"):
            layout_prefix = task_name.split("/", 1)[1]  # 'coord_ring' etc.
        elif task_name.startswith("lbf/"):
            # LBF dirs may use 'lbf_7x7_nolevels' as a subdir prefix; try it
            layout_prefix = task_name.split("/", 1)[1]
        log.info("loading BRs for %d partners from %s (layout_prefix=%s)",
                 len(agents), br_root, layout_prefix)
        brs = load_brs_for_pd(env, env_kwargs, task_name, br_root, list(agents.keys()),
                               seed, layout_prefix=layout_prefix)
        log.info("loaded %d BRs (%d partners will fall back to self-play)",
                 len(brs), len(agents) - len(brs))

    log.info("computing PD for %s with %d agents x %d episodes (br_paired=%s)",
             env_cfg.name, len(agents), num_episodes, br_paired)

    theta_rows: List[np.ndarray] = []
    agent_names: List[str] = []
    per_agent_records: List[Dict[str, Any]] = []
    pairing_modes: List[str] = []

    for agent_name, (policy, params) in agents.items():
        partner_label = "BR" if (br_paired and agent_name in brs) else "self-play"
        pairing_modes.append(partner_label)
        log.info("  rollouts: %s (%s)", agent_name, partner_label)
        if br_paired and agent_name in brs:
            br_policy, br_params = brs[agent_name]
            ep_counts = env_cfg.rollout_two_policy(
                env, policy, params, br_policy, br_params, layout, num_episodes, seed,
            )
        else:
            ep_counts = env_cfg.rollout(env, policy, layout, num_episodes, seed, params=params)
        ep_vectors = np.stack(
            [env_cfg.to_vector(c, env_cfg.score_norm, env_cfg.length_norm) for c in ep_counts]
        )
        theta = ep_vectors.mean(axis=0)
        theta_rows.append(theta)
        agent_names.append(agent_name)

        per_agent_records.append(
            dict(
                agent=agent_name,
                env=env_cfg.name,
                pairing=partner_label,
                num_episodes=num_episodes,
                **{
                    name: float(theta[i]) for i, name in enumerate(env_cfg.feature_names)
                },
                mean_final_score=float(np.mean([c.final_score for c in ep_counts])),
                mean_episode_length=float(np.mean([c.episode_length for c in ep_counts])),
            )
        )

    theta = np.stack(theta_rows)
    if len(set(pairing_modes)) > 1:
        n_br = pairing_modes.count("BR")
        log.warning(
            "PD theta mixes %d BR rows with %d self-play rows in one det(K); they are not comparable "
            "(missing BRs fell back to self-play). See the per-row 'pairing' field.",
            n_br, len(pairing_modes) - n_br,
        )
    pd_payload = population_diversity(theta)

    csv_path = output_dir / "features.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(per_agent_records[0].keys()))
        writer.writeheader()
        writer.writerows(per_agent_records)

    json_payload = {
        "env": env_cfg.name,
        "num_agents": len(agent_names),
        "num_episodes": num_episodes,
        "seed": seed,
        "feature_names": env_cfg.feature_names,
        "agent_names": agent_names,
        "pairing": pairing_modes,
        "theta": theta.tolist(),
        "pd": pd_payload,
    }
    with (output_dir / "pd.json").open("w") as fh:
        json.dump(json_payload, fh, indent=2)

    # PCA on the SAME normalized theta that K/cosine/heatmap use, so every
    # downstream figure is consistent with the det(K) metric. Raw theta stays
    # available in pd.json (pd_payload['theta_raw']) and the top-level 'theta'.
    save_pca_plot(
        np.array(pd_payload["theta_norm"]),
        agent_names,
        output_dir / f"pca_{env_cfg.name}.pdf",
        f"PCA of normalized theta vectors -- {env_cfg.name}",
    )
    save_cosine_heatmap(
        np.array(pd_payload["cosine"]),
        agent_names,
        output_dir / f"heatmap_{env_cfg.name}.pdf",
        f"Pairwise cosine similarity -- {env_cfg.name}",
    )

    log.info(
        "  PD = det(K) = %.6e  (log det = %.4f, sign = %d)  -- %s",
        pd_payload["det_K"],
        pd_payload["log_det_K"],
        pd_payload["sign"],
        env_cfg.name,
    )
    return json_payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute Population Diversity for the heuristic teammate suite.")
    parser.add_argument(
        "--env",
        choices=["hanabi", "mini-hanabi", "lbf", "overcooked", "all"],
        default="mini-hanabi",
        help="Environment to compute PD for. 'all' iterates over hanabi+mini-hanabi+lbf+overcooked.",
    )
    parser.add_argument(
        "--variant",
        default=None,
        help=(
            "Layout variant for lbf or overcooked. "
            "lbf: lbf_7x7_nolevels (default) or lbf_12x12. "
            "overcooked: cramped_room (default), coord_ring, etc. Ignored for hanabi/mini-hanabi."
        ),
    )
    parser.add_argument("--num-episodes", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default="results/population_diversity",
        help="Per-env subdirectories are created under this root.",
    )
    parser.add_argument(
        "--full-heldout",
        action="store_true",
        help="Use the full eval set from evaluation/configs/global_heldout_settings.yaml (heuristics + RL partners + BC + OBL) instead of the heuristic-only suite.",
    )
    parser.add_argument(
        "--br-paired",
        action="store_true",
        help=(
            "Pair each partner with its trained best-response (canonical Wang et al. 2024 PD). "
            "Requires --br-root pointing at a directory tree of saved_train_run BR checkpoints. "
            "If a BR is missing for a partner, that partner falls back to self-play with a logged warning."
        ),
    )
    parser.add_argument(
        "--br-root",
        default=None,
        help=(
            "Root directory for BR checkpoints. Expected layout: <br_root>/<partner_name>/saved_train_run/. "
            "Typical setup: download jaxaht/eval-teammates-br/<env>/ from HF and point --br-root at the local copy."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    env_targets: List[str]
    if args.env == "all":
        env_targets = ["hanabi", "mini-hanabi", "lbf", "overcooked"]
    else:
        env_targets = [args.env]

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.br_paired and not args.full_heldout:
        log.warning("--br-paired implies --full-heldout (need a partner set to BR against); enabling --full-heldout.")
        args.full_heldout = True
    if args.br_paired and not args.br_root:
        raise ValueError("--br-paired requires --br-root pointing at the local BR checkpoint root.")

    br_root = Path(args.br_root) if args.br_root else None

    summary: Dict[str, Any] = {"runs": [], "full_heldout": args.full_heldout, "br_paired": args.br_paired}
    for env_name in env_targets:
        try:
            if env_name in HANABI_VARIANTS:
                env_cfg = env_config_for_hanabi(env_name)
                yaml_key = env_name
                task_name = env_name
                out_subdir = env_name
            elif env_name == "lbf":
                variant = args.variant or "lbf_7x7_nolevels"
                env_cfg = env_config_for_lbf(variant)
                yaml_key = f"lbf/{variant}"
                task_name = f"lbf/{variant}"
                out_subdir = f"lbf-{variant}"
            elif env_name == "overcooked":
                variant = args.variant or "cramped_room"
                env_cfg = env_config_for_overcooked(variant)
                yaml_key = f"overcooked-v1/{variant}"
                task_name = f"overcooked-v1/{variant}"
                out_subdir = f"overcooked-{variant}"
            else:
                raise ValueError(env_name)
            if args.full_heldout:
                out_subdir = f"{out_subdir}_full"
            if args.br_paired:
                out_subdir = f"{out_subdir}_brpaired"
            payload = compute_pd_for_env(
                env_cfg,
                args.num_episodes,
                args.seed,
                output_root / out_subdir,
                full_heldout=args.full_heldout,
                heldout_yaml_key=yaml_key if args.full_heldout else None,
                task_name=task_name if args.full_heldout else None,
                br_paired=args.br_paired,
                br_root=br_root,
            )
            summary["runs"].append({"env": env_name, "pd": payload["pd"]["det_K"], "n_agents": payload["num_agents"]})
        except NotImplementedError as exc:
            log.warning("skipping %s: %s", env_name, exc)
            summary["runs"].append({"env": env_name, "pd": None, "skipped_reason": str(exc)})
        except Exception as exc:
            log.exception("failed %s: %s", env_name, exc)
            summary["runs"].append({"env": env_name, "pd": None, "error": str(exc)[:200]})

    with (output_root / "summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)

    print("\nPD summary:")
    for entry in summary["runs"]:
        if entry["pd"] is None:
            print(f"  {entry['env']}: skipped ({entry.get('skipped_reason', 'no reason given')[:80]}...)")
        else:
            print(f"  {entry['env']}: det(K) = {entry['pd']:.6e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

