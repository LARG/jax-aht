# Population Diversity CLI: rollouts -> theta features -> PD, features.csv, pd.json, heatmap + PCA.
# Example: python scripts/population_diversity/compute_population_diversity.py --env overcooked \
#   --variant coord_ring --full-heldout --br-paired --br-root <br_root> --batched --cache-dir results/pd_cache
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from envs import make_env
from envs.log_wrapper import LogWrapper

from scripts.population_diversity.pd_events import (
    HANABI_FEATURE_NAMES,
    LBF_FEATURE_NAMES,
    hanabi_feature_names,
    OVERCOOKED_CONTINGENT_FEATURE_NAMES,
    OVERCOOKED_FEATURE_NAMES,
    OVERCOOKED_V5_FEATURE_NAMES,
)
from scripts.population_diversity.pd_plots import (
    population_diversity,
    pca_2d,
    save_pca_plot,
    save_cosine_heatmap,
)
from scripts.population_diversity.pd_rollouts import (
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
    build_overcooked_agents,
    load_full_heldout_for_pd,
    load_brs_for_pd,
    rollout_two_policy_batched,
)
from scripts.population_diversity.pd_traj_cache import (
    OvercookedTrajRecorder,
    save_batched_traj,
    cache_path,
    load_episodes,
    replay_counts,
)
from scripts.population_diversity.pd_overcooked_features import (
    OvercookedEpisodeCounts,
    episode_contingency_features,
    episode_v5_features,
    overcooked_cell_support,
    overcooked_episode_to_vector,
    overcooked_step_update,
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
    rollout: Callable                            # self-play rollout
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

    def _horizon(env):
        return min(int(getattr(getattr(env._env, "env", None), "time_limit", 128)), 128)

    def _rollout(env, policy, layout, num_eps, seed, params=None, recorder=None):
        return rollout_simple_self_play(
            env, policy, num_eps, seed, LBFEpisodeCounts, partial(lbf_step_update, horizon=_horizon(env)),
            max_steps=128, params=params, recorder=recorder,
        )

    def _rollout_two_policy(env, policy_a, params_a, policy_b, params_b, layout, num_eps, seed,
                            recorder=None):
        return rollout_two_policy(
            env, policy_a, params_a, policy_b, params_b,
            num_eps, seed, LBFEpisodeCounts, partial(lbf_step_update, horizon=_horizon(env)),
            max_steps=128, recorder=recorder,
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



def env_config_for_overcooked(layout_name: str = "cramped_room",
                              deterministic_reset: bool = False) -> EnvConfig:
    from envs.overcooked.augmented_layouts import augmented_layouts

    layout = augmented_layouts[layout_name]

    def _build_env():
        env_kwargs = {
            "layout": layout_name,
            "random_obj_state": not deterministic_reset,
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
        if deterministic_reset:
            # Fixed start cells; agent_idx is swapped below so the tracked teammate (agent_1) takes agent_0's cell.
            env_kwargs["random_reset"] = False
        env = make_env("overcooked-v1", env_kwargs)
        if deterministic_reset:
            from flax.core.frozen_dict import FrozenDict
            base = env.env  # unwrap to the jaxmarl Overcooked env
            idx = jnp.asarray(base.layout["agent_idx"])[::-1]
            base.layout = FrozenDict({**dict(base.layout), "agent_idx": idx})
        env = LogWrapper(env)
        return env, env_kwargs

    def _build_agents(env, env_kwargs):
        agents = build_overcooked_agents(layout=layout)
        return agents, None

    def _rollout(env, policy, layout_obj, num_eps, seed, params=None, recorder=None):
        return rollout_simple_self_play(
            env, policy, num_eps, seed, OvercookedEpisodeCounts, overcooked_step_update,
            max_steps=400, params=params, recorder=recorder,
        )

    def _rollout_two_policy(env, policy_a, params_a, policy_b, params_b, layout_obj, num_eps, seed,
                            recorder=None):
        return rollout_two_policy(
            env, policy_a, params_a, policy_b, params_b,
            num_eps, seed, OvercookedEpisodeCounts, overcooked_step_update, max_steps=400,
            recorder=recorder,
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



def _theta_from_counts(env_cfg: EnvConfig, ep_counts) -> np.ndarray:
    ep_vectors = np.stack(
        [env_cfg.to_vector(c, env_cfg.score_norm, env_cfg.length_norm) for c in ep_counts]
    )
    return ep_vectors.mean(axis=0)


def _record_from_counts(env_cfg: EnvConfig, agent_name: str, num_episodes: int, ep_counts,
                        extra: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    theta = _theta_from_counts(env_cfg, ep_counts)
    return dict(
        agent=agent_name,
        env=env_cfg.name,
        num_episodes=num_episodes,
        **{name: float(theta[i]) for i, name in enumerate(env_cfg.feature_names)},
        **(extra or {}),
        mean_final_score=float(np.mean([c.final_score for c in ep_counts])),
        mean_episode_length=float(np.mean([c.episode_length for c in ep_counts])),
    )


def _write_outputs(env_cfg: EnvConfig, agent_names, per_agent_records, theta_rows,
                   num_episodes: int, seed: int, output_dir: Path,
                   support_rows: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Write features.csv, cell_support.csv, pd.json and the PCA / cosine-heatmap PDFs."""
    theta_mat = np.stack(theta_rows)
    pd_payload = population_diversity(theta_mat)

    csv_path = output_dir / "features.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(per_agent_records[0].keys()))
        writer.writeheader()
        writer.writerows(per_agent_records)
    if support_rows:
        with (output_dir / "cell_support.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(support_rows[0].keys()))
            writer.writeheader()
            writer.writerows(support_rows)

    json_payload = {
        "env": env_cfg.name,
        "num_agents": len(agent_names),
        "num_episodes": num_episodes,
        "seed": seed,
        "feature_names": env_cfg.feature_names,
        "agent_names": agent_names,
        "theta": theta_mat.tolist(),
        "pd": pd_payload,
    }
    with (output_dir / "pd.json").open("w") as fh:
        json.dump(json_payload, fh, indent=2)

    save_pca_plot(
        np.array(pd_payload["theta_norm"]), agent_names,
        output_dir / f"pca_{env_cfg.name}.pdf",
        f"PCA of normalized theta vectors -- {env_cfg.name}",
    )
    save_cosine_heatmap(
        np.array(pd_payload["cosine"]), agent_names,
        output_dir / f"heatmap_{env_cfg.name}.pdf",
        f"Pairwise cosine similarity -- {env_cfg.name}",
    )
    log.info(
        "  PD = det(K) = %.6e  (log det = %.4f, sign = %d)  -- %s",
        pd_payload["det_K"], pd_payload["log_det_K"], pd_payload["sign"], env_cfg.name,
    )
    return json_payload


def _agent_sort_key(stem: str):
    """Numeric sort so `agent_7` precedes `agent_11`."""
    import re as _re
    return [int(x) if x.isdigit() else x for x in _re.split(r"(\d+)", stem)]


def _features_from_cache_file(env_cfg: EnvConfig, f: Path, contingent: bool = True,
                              v5: bool = True):
    """(agent_name, ep_counts, features record, cell-support row) for one cached teammate."""
    eps = load_episodes(f)
    ep_counts = replay_counts(f)
    agent_name = str(np.load(f, allow_pickle=False)["agent"])
    extra: Dict[str, float] = {}
    if contingent:
        per_ep = np.stack([episode_contingency_features(e) for e in eps])
        extra = {name: float(per_ep[:, i].mean())
                 for i, name in enumerate(OVERCOOKED_CONTINGENT_FEATURE_NAMES)}
    if v5:
        per_ep5 = np.stack([episode_v5_features(e) for e in eps])
        extra.update({name: float(per_ep5[:, i].mean())
                      for i, name in enumerate(OVERCOOKED_V5_FEATURE_NAMES)})
    record = _record_from_counts(env_cfg, agent_name, len(ep_counts), ep_counts, extra=extra)
    support = {"agent": agent_name, **overcooked_cell_support(ep_counts)}
    return agent_name, ep_counts, record, support


def compute_pd_from_cache(
    env_cfg: EnvConfig,
    cache_dir: Path,
    output_dir: Path,
    seed: int = 0,
    contingent: bool = True,
    v5: bool = True,
) -> Dict[str, Any]:
    """Recompute features from a trajectory cache on CPU."""
    output_dir.mkdir(parents=True, exist_ok=True)
    task_dir = Path(cache_dir) / env_cfg.name.replace("/", "_")
    files = sorted(task_dir.glob("*.npz"), key=lambda f: _agent_sort_key(f.stem))
    if not files:
        raise FileNotFoundError(f"no cache files under {task_dir}")
    log.info("recomputing features from %d cached teammates in %s", len(files), task_dir)

    agent_names: List[str] = []
    theta_rows: List[np.ndarray] = []
    records: List[Dict[str, Any]] = []
    support_rows: List[Dict[str, Any]] = []
    for f in files:
        agent_name, ep_counts, record, support = _features_from_cache_file(env_cfg, f, contingent, v5)
        support_rows.append(support)
        records.append(record)
        agent_names.append(agent_name)
        theta_rows.append(_theta_from_counts(env_cfg, ep_counts))
        log.info("  %s: %d episodes", agent_name, len(ep_counts))

    num_episodes = int(np.median([r["num_episodes"] for r in records]))
    return _write_outputs(env_cfg, agent_names, records, theta_rows,
                          num_episodes, seed, output_dir, support_rows=support_rows)


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
    cache_dir: Path | None = None,
    batched: bool = False,
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
        if env_cfg.name in HANABI_VARIANTS:
            # Hanabi needs the action layout to decode actions; LBF/Overcooked use None.
            cfg = HANABI_VARIANTS[env_cfg.name]
            layout = HanabiActionLayout(
                hand_size=cfg["hand_size"], num_colors=cfg["num_colors"],
                num_ranks=cfg["num_ranks"], num_actions=env.action_space("agent_0").n,
            )
        else:
            layout = None
    else:
        agents_dict, layout = env_cfg.build_agents(env, env_kwargs)
        agents = {name: (policy, None) for name, policy in agents_dict.items()}

    # BR-paired: theta is attributed to the teammate (agent_1), BR is agent_0; missing BR -> self-play.
    brs: Dict[str, Tuple[Any, Any]] = {}
    if br_paired:
        if br_root is None:
            raise ValueError("br_paired requires br_root")
        if task_name is None:
            raise ValueError("br_paired requires task_name (e.g. 'mini-hanabi', 'lbf/lbf_7x7_nolevels', 'overcooked-v1/counter_circuit')")
        # layout_prefix for HF-style BR checkpoint naming
        layout_prefix = None
        if task_name.startswith("overcooked-v1/"):
            layout_prefix = task_name.split("/", 1)[1]
        elif task_name.startswith("lbf/"):
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
    support_rows: List[Dict[str, Any]] = []
    pairing_modes: List[str] = []

    is_overcooked = env_cfg.name.startswith("overcooked")
    if (cache_dir is not None or batched) and not is_overcooked:
        raise NotImplementedError(
            "the trajectory cache and the batched rollout are implemented for Overcooked "
            "only; LBF still recomputes features inside the sequential rollout loop."
        )
    if batched and cache_dir is None:
        raise ValueError("--batched requires --cache-dir (the batched output IS the cache)")

    for agent_name, (policy, params) in agents.items():
        if br_paired and agent_name in brs:
            pairing = "BR"
        else:
            pairing = "self-play"
        pairing_modes.append(pairing)
        if batched:
            # Slot assignment mirrors the sequential branches below exactly.
            if pairing == "BR":
                a0, a1 = (policy, params), brs[agent_name]
            else:
                a0, a1 = (policy, params), (policy, params)
            log.info("  batched rollout: %s (%s)", agent_name, pairing)
            t0 = time.time()
            traj = rollout_two_policy_batched(
                env, a0[0], a0[1], a1[0], a1[1], num_episodes, seed, max_steps=400,
            )
            cpath = cache_path(cache_dir, env_cfg.name, agent_name)
            nbytes = save_batched_traj(traj, agent_name, env_cfg.name, seed, cpath)
            del traj
            log.info("    %.1fs wall; cache %s (%.1f MB)", time.time() - t0, cpath, nbytes / 1e6)
            _, ep_counts, record, support = _features_from_cache_file(env_cfg, cpath)
            record["pairing"] = pairing
            theta_rows.append(_theta_from_counts(env_cfg, ep_counts))
            agent_names.append(agent_name)
            per_agent_records.append(record)
            support_rows.append(support)
            continue

        recorder = OvercookedTrajRecorder(agent_name, env_cfg.name, seed) if cache_dir else None
        log.info("  rollouts: %s (%s)", agent_name, pairing)
        if pairing == "BR":
            # Heldout partner = agent_0, its BR = agent_1 (ZSC-Eval slot order).
            br_policy, br_params = brs[agent_name]
            ep_counts = env_cfg.rollout_two_policy(
                env, policy, params, br_policy, br_params, layout, num_episodes, seed,
                recorder=recorder,
            )
        else:
            ep_counts = env_cfg.rollout(env, policy, layout, num_episodes, seed, params=params,
                                        recorder=recorder)
        if recorder is not None:
            cpath = cache_path(cache_dir, env_cfg.name, agent_name)
            nbytes = recorder.save(cpath)
            log.info("    cached trajectories -> %s (%.1f MB)", cpath, nbytes / 1e6)
        theta_rows.append(_theta_from_counts(env_cfg, ep_counts))
        agent_names.append(agent_name)
        per_agent_records.append(
            _record_from_counts(env_cfg, agent_name, num_episodes, ep_counts,
                                extra={"pairing": pairing}))
        if is_overcooked:
            support_rows.append({"agent": agent_name, **overcooked_cell_support(ep_counts)})

    if len(set(pairing_modes)) > 1:
        log.warning(
            "PD theta mixes pairing modes %s in one det(K); rows are not comparable "
            "(missing BRs fell back to self-play). See the per-row 'pairing' field.",
            {m: pairing_modes.count(m) for m in set(pairing_modes)},
        )
    return _write_outputs(env_cfg, agent_names, per_agent_records, theta_rows,
                          num_episodes, seed, output_dir, support_rows=support_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute Population Diversity for the heuristic teammate suite.")
    parser.add_argument(
        "--env",
        choices=["hanabi", "mini-hanabi", "lbf", "overcooked", "all"],
        default="mini-hanabi",
        help="Environment; 'all' runs hanabi, mini-hanabi, lbf and overcooked.",
    )
    parser.add_argument(
        "--variant",
        default=None,
        help="Variant for the chosen env: lbf_7x7_nolevels (default) / lbf_12x12; overcooked layouts. Ignored for hanabi.",
    )
    parser.add_argument("--num-episodes", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default="results/population_diversity",
        help="Root output directory.",
    )
    parser.add_argument(
        "--full-heldout",
        action="store_true",
        help="Use the full heldout eval set instead of the heuristic suite.",
    )
    parser.add_argument(
        "--br-paired",
        action="store_true",
        help="Pair each partner with its best response (requires --br-root).",
    )
    parser.add_argument(
        "--br-root",
        default=None,
        help="Root of BR checkpoints: <br_root>/<partner>/saved_train_run/.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Write per-step transitions to <cache-dir>/<task>/<agent>.npz (Overcooked only).",
    )
    parser.add_argument(
        "--batched",
        action="store_true",
        help="Use the vmapped batched rollout (requires --cache-dir).",
    )
    parser.add_argument(
        "--from-cache",
        action="store_true",
        help="Recompute features from an existing --cache-dir without rollouts.",
    )
    parser.add_argument(
        "--deterministic-reset",
        action="store_true",
        help="Overcooked only: fixed start cells and empty pots.",
    )
    parser.add_argument(
        "--no-contingent",
        action="store_true",
        help="With --from-cache, omit the partner-contingent features.",
    )
    parser.add_argument(
        "--no-v5",
        action="store_true",
        help="With --from-cache, omit the contention and action-MI features.",
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
                env_cfg = env_config_for_overcooked(
                    variant, deterministic_reset=args.deterministic_reset)
                yaml_key = f"overcooked-v1/{variant}"
                task_name = f"overcooked-v1/{variant}"
                out_subdir = f"overcooked-{variant}"
            else:
                raise ValueError(env_name)
            if args.full_heldout:
                out_subdir = f"{out_subdir}_full"
            if args.br_paired:
                out_subdir = f"{out_subdir}_brpaired"
            if args.from_cache:
                if not args.cache_dir:
                    raise ValueError("--from-cache requires --cache-dir")
                payload = compute_pd_from_cache(
                    env_cfg, Path(args.cache_dir), output_root / out_subdir,
                    seed=args.seed, contingent=not args.no_contingent,
                    v5=not args.no_v5,
                )
                summary["runs"].append({"env": env_name, "pd": payload["pd"]["det_K"],
                                        "n_agents": payload["num_agents"]})
                continue
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
                cache_dir=Path(args.cache_dir) if args.cache_dir else None,
                batched=args.batched,
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

