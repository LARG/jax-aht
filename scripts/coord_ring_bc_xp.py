"""Cross-play matrix: BC human-proxy × {ppo_ego, liam_ego, meliba_ego} on a
chosen overcooked-v1 layout. The runs evaluated for each (algo, teammate_set)
cell are hardcoded in RUNS_BY_LAYOUT below — discovery is deliberately external
(curate manually, paste the wandb ids in).

Evaluates BOTH agent positions (ego at slot 0 vs BC at slot 0).
Sweeps over all 5 BC checkpoints (run_id 0..4) per ego variant.

Vmap layout:
- Per (algo, position, bc_run_id): one vmapped call over (run × seed) variants ×
  episode rngs. All variants of the same algo share the actor architecture, so
  params are concatenated along the leading (seed) axis across runs and become
  a flat variant batch.

Usage:
  /scratch/cluster/jyliu/conda_envs/HANABI/bin/python scripts/coord_ring_bc_xp.py [--layout LAYOUT]
"""
import argparse
import csv
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from agents.initialize_agents import (
    initialize_liam_agent,
    initialize_meliba_agent,
    initialize_s5_agent,
)
from agents.overcooked.bc_agent import BCPolicy
from common.save_load_utils import load_train_run
from envs import make_env
from envs.log_wrapper import LogWrapper

NUM_EPS = 10
EVAL_SEED = 34957
ROLLOUT_LEN = 400
BC_RUN_IDS = (0, 1, 2, 3, 4)

# ---- Per-layout settings ----
# do_reward_shaping=True gives both shaped + base in the env step info; the
# bar chart plots whichever metric matches the env's training reward (so coord_ring
# uses shaped, all other layouts use sparse/base since their evals run unshaped).
SHAPING_PARAMS = {
    "PLACEMENT_IN_POT_REW": 0.5,
    "PLATE_PICKUP_REWARD": 0.1,
    "SOUP_PICKUP_REWARD": 1.0,
    "ONION_PICKUP_REWARD": 0.1,
    "COUNTER_PICKUP_REWARD": 0,
    "COUNTER_DROP_REWARD": 0,
}
LAYOUT_CFG = {
    "coord_ring": {
        "do_reward_shaping": True,
        "plot_metric": "shaped",       # "shaped" or "base"
        "bc_selfplay_ref": 268.481,    # shaped BC×BC self-play (run_id=0)
    },
    "cramped_room": {
        "do_reward_shaping": False,
        "plot_metric": "base",
        "bc_selfplay_ref": 196.25,     # base BC×BC self-play (run_id=0)
    },
    "asymm_advantages": {
        "do_reward_shaping": False,
        "plot_metric": "base",
        "bc_selfplay_ref": 371.25,
    },
    "counter_circuit": {
        "do_reward_shaping": False,
        "plot_metric": "base",
        "bc_selfplay_ref": 120.63,
    },
    "forced_coord": {
        "do_reward_shaping": False,
        "plot_metric": "base",
        "bc_selfplay_ref": 152.50,
    },
}


def build_env_kwargs(layout):
    cfg = LAYOUT_CFG[layout]
    kw = {"layout": layout, "random_obj_state": True,
          "do_reward_shaping": cfg["do_reward_shaping"]}
    if cfg["do_reward_shaping"]:
        kw["reward_shaping_params"] = SHAPING_PARAMS
    return kw

PPO_EGO_CFG = {
    "EGO_ACTOR_TYPE": "s5",
    "S5_D_MODEL": 128,
    "S5_SSM_SIZE": 128,
    "S5_N_LAYERS": 2,
    "S5_BLOCKS": 1,
    "S5_ACTOR_CRITIC_HIDDEN_DIM": 1024,
    "FC_N_LAYERS": 3,
    "S5_ACTIVATION": "full_glu",
    "S5_DO_NORM": True,
    "S5_PRENORM": True,
    "S5_DO_GTRXL_NORM": True,
}

LIAM_CFG = {
    "EGO_ACTOR_TYPE": "mlp",
    "FC_HIDDEN_DIM": 64,
    "ENCODER_TYPE": "lstm",
    "ENCODER_HIDDEN_DIM": 64,
    "ENCODER_OUTPUT_DIM": 20,
    "DECODER_HIDDEN_DIM": 64,
}

MELIBA_CFG = {
    "EGO_ACTOR_TYPE": "mlp",
    "FC_HIDDEN_DIM": 64,
    "ENCODER_STATE_EMBED_DIM": 64,
    "ENCODER_ACTION_EMBED_DIM": 64,
    "ENCODER_REWARD_EMBED_DIM": 64,
    "ENCODER_RNN_HIDDEN_DIM": 64,
    "ENCODER_LAYERS_BEFORE_RNN": 64,
    "ENCODER_LATENT_DIM": 64,
}

# Each (algo) -> list of (wandb_id, teammate_set). Multiple entries with the same
# teammate set are pooled along the leading seed axis (e.g., a 3-seed + 2-seed
# meliba pair forms a 5-variant cell). Curated manually — see RUNS_INVENTORY.md
# for how these were discovered (project=aht-project/aht-benchmark,
# tags=neurips:benchmark + neurips:benchmark:<set>_teammates + overcooked-v1/<layout>).
RUNS_BY_LAYOUT = {
    "coord_ring": {
        "ppo_ego":    [("0wzr6nbv", "fcp"), ("7mrfwnra", "rotate"), ("5njknt2q", "comedi")],
        "liam_ego":   [("e8y4gy49", "fcp"), ("4vdq0mjv", "rotate"), ("pmm9u33p", "comedi")],
        "meliba_ego": [("x6gmp9uc", "fcp"), ("c9o32dg3", "rotate"),
                       ("bd6w70tr", "comedi"), ("7a19mtio", "comedi")],
    },
    "cramped_room": {
        "ppo_ego":    [("rm5bx4ui", "fcp"), ("7jhrdpxc", "comedi")],
        "liam_ego":   [("xuvmlpmi", "fcp"), ("8486vdnp", "comedi")],
        "meliba_ego": [("v5ey4lkt", "fcp"), ("lghgomp3", "fcp"),
                       ("3e6wffoi", "comedi"), ("en1wbqt4", "comedi")],
    },
    # asymm_advantages / counter_circuit / forced_coord intentionally absent —
    # no ppo_ego/liam_ego/meliba_ego wandb runs exist for those layouts (only
    # teammate-generation runs: brdiv/comedi/fcp/ippo/lbrdiv/ppo_br/rotate).
}


def build_env(layout):
    return LogWrapper(make_env("overcooked-v1", build_env_kwargs(layout)))


def build_ego_policy(algo, env, init_rng):
    if algo == "ppo_ego":
        return initialize_s5_agent(PPO_EGO_CFG, env, init_rng)
    if algo == "liam_ego":
        cfg = dict(LIAM_CFG)
        cfg["POLICY_INPUT_DIM"] = (
            cfg["ENCODER_OUTPUT_DIM"]
            + env.observation_space(env.agents[0]).shape[0]
        )
        return initialize_liam_agent(cfg, env, init_rng)
    if algo == "meliba_ego":
        cfg = dict(MELIBA_CFG)
        cfg["POLICY_INPUT_DIM"] = (
            cfg["ENCODER_LATENT_DIM"] * 4
            + env.observation_space(env.agents[0]).shape[0]
        )
        return initialize_meliba_agent(cfg, env, init_rng)
    raise ValueError(algo)


def extract_ego_params(algo, final):
    """Return the per-seed param pytree expected by the policy (leading axis = seeds)."""
    if algo == "ppo_ego":
        return final  # {'params': ...}
    return {
        "encoder": final["encoder"],
        "decoder": final["decoder"],
        "policy":  final["policy"],
    }


def rollout_single_ep(
    rng, env, p0_pol, p0_param, p0_test_mode, p1_pol, p1_param, p1_test_mode, max_steps
):
    """Like common.run_episodes.run_single_episode but feeds aux_obs to BOTH agents."""
    rng, reset_rng = jax.random.split(rng)
    obs, env_state = env.reset(reset_rng)
    init_done = {k: jnp.zeros((1), dtype=bool) for k in env.agents + ["__all__"]}
    init_act_oh = {
        k: jnp.zeros((env.action_space(env.agents[i]).n))
        for i, k in enumerate(env.agents)
    }
    init_reward = {k: jnp.zeros((1)) for k in env.agents}
    init_joint = jnp.concatenate(
        (init_act_oh["agent_0"].reshape(1, 1, -1),
         init_act_oh["agent_1"].reshape(1, 1, -1)),
        axis=-1,
    )

    h0 = p0_pol.init_hstate(1, aux_info={"agent_id": 0})
    h1 = p1_pol.init_hstate(1, aux_info={"agent_id": 1})

    def call(pol, param, obs_a, done_a, avail_a, h, rng_a, act_oh_a, joint, reward_a,
             env_state, test_mode):
        return pol.get_action(
            params=param,
            obs=obs_a.reshape(1, 1, -1),
            done=done_a.reshape(1, 1),
            avail_actions=avail_a.astype(jnp.float32),
            hstate=h, rng=rng_a,
            aux_obs=(act_oh_a.reshape(1, 1, -1), joint, reward_a.reshape(1, 1, -1)),
            env_state=env_state, test_mode=test_mode,
        )

    avail = env.get_avail_actions(env_state)
    rng, k0, k1, ks = jax.random.split(rng, 4)
    a0, h0 = call(p0_pol, p0_param, obs["agent_0"], init_done["agent_0"], avail["agent_0"],
                  h0, k0, init_act_oh["agent_0"], init_joint, init_reward["agent_0"],
                  env_state, p0_test_mode)
    a1, h1 = call(p1_pol, p1_param, obs["agent_1"], init_done["agent_1"], avail["agent_1"],
                  h1, k1, init_act_oh["agent_1"], init_joint, init_reward["agent_1"],
                  env_state, p1_test_mode)
    a0 = a0.squeeze(); a1 = a1.squeeze()
    env_act = {"agent_0": a0, "agent_1": a1}
    env_act_oh = {k: jax.nn.one_hot(env_act[k], env.action_space(env.agents[i]).n)
                  for i, k in enumerate(env.agents)}
    obs, env_state, reward, done, info = env.step(ks, env_state, env_act)

    init_carry = (1, env_state, obs, rng, done, reward, env_act_oh, h0, h1, info)

    def scan_step(carry, _):
        def take(carry):
            ts, env_state, obs, rng, done, reward, act_oh, h0, h1, _last = carry
            avail = env.get_avail_actions(env_state)
            joint = jnp.concatenate(
                (act_oh["agent_0"].reshape(1, 1, -1),
                 act_oh["agent_1"].reshape(1, 1, -1)),
                axis=-1,
            )
            rng, k0, k1, ks = jax.random.split(rng, 4)
            a0, h0n = call(p0_pol, p0_param, obs["agent_0"], done["agent_0"], avail["agent_0"],
                           h0, k0, act_oh["agent_0"], joint, reward["agent_0"],
                           env_state, p0_test_mode)
            a1, h1n = call(p1_pol, p1_param, obs["agent_1"], done["agent_1"], avail["agent_1"],
                           h1, k1, act_oh["agent_1"], joint, reward["agent_1"],
                           env_state, p1_test_mode)
            a0 = a0.squeeze(); a1 = a1.squeeze()
            env_act = {"agent_0": a0, "agent_1": a1}
            env_act_oh = {k: jax.nn.one_hot(env_act[k], env.action_space(env.agents[i]).n)
                          for i, k in enumerate(env.agents)}
            obs_n, env_state_n, reward_n, done_n, info_n = env.step(ks, env_state, env_act)
            return (ts + 1, env_state_n, obs_n, rng, done_n, reward_n, env_act_oh,
                    h0n, h1n, info_n)
        return jax.lax.cond(carry[4]["__all__"], lambda c: c, take, carry), None

    final, _ = jax.lax.scan(scan_step, init_carry, None, length=max_steps)
    return final[-1]


def run_for_algo(layout, algo, env, ego_policy, bc_policies, run_specs, rng_master):
    """Run all (run, seed, position, bc_run_id, episode) for one algorithm.

    One vmapped pass per (position, bc_run_id) — variant batch is (V × NUM_EPS).

    Args:
        layout: overcooked-v1 layout name (used to resolve wandb_<id> path).
        bc_policies: dict mapping bc_run_id -> BCPolicy instance.
        run_specs: list of (run_id, teammate_set) for this algo.
    Returns:
        list of dicts: {run_id, teammate_set, position, bc_run_id, seed_idx,
                        shaped_per_ep, base_per_ep}
    """
    # Load all runs' params and stack to a flat variant batch.
    per_run_meta = []
    per_run_params = []
    for rid, tt in run_specs:
        path = f"results/overcooked-v1/{layout}/{algo}/wandb_{rid}"
        out = load_train_run(path)
        params = extract_ego_params(algo, out["final_params"])
        n_seeds = jax.tree.leaves(params)[0].shape[0]
        per_run_meta.append((rid, tt, n_seeds))
        per_run_params.append(params)

    # Concatenate along leading (seed) axis → flat (V, ...) where V = sum of all n_seeds.
    flat_params = jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *per_run_params)
    V = jax.tree.leaves(flat_params)[0].shape[0]
    print(f"  total variants V={V} (runs: {[(m[0], m[2]) for m in per_run_meta]})", flush=True)

    # Build variant_idx → (run_id, teammate_set, seed_idx) labels.
    labels = []
    for rid, tt, ns in per_run_meta:
        for s in range(ns):
            labels.append((rid, tt, s))
    assert len(labels) == V

    results = []
    for position in ("ego_first", "bc_first"):
        for bc_run_id in BC_RUN_IDS:
            bc_policy = bc_policies[bc_run_id]
            rng_master, rng_pos = jax.random.split(rng_master)
            per_variant_rngs = jax.random.split(rng_pos, V)

            if position == "ego_first":
                p0_pol, p1_pol = ego_policy, bc_policy
                p0_test, p1_test = False, True

                def per_variant(rng, ego_p):
                    rngs = jax.random.split(rng, NUM_EPS)
                    return jax.vmap(
                        lambda r: rollout_single_ep(
                            r, env, p0_pol, ego_p, p0_test,
                            p1_pol, None, p1_test, ROLLOUT_LEN,
                        )
                    )(rngs)
            else:
                p0_pol, p1_pol = bc_policy, ego_policy
                p0_test, p1_test = True, False

                def per_variant(rng, ego_p):
                    rngs = jax.random.split(rng, NUM_EPS)
                    return jax.vmap(
                        lambda r: rollout_single_ep(
                            r, env, p0_pol, None, p0_test,
                            p1_pol, ego_p, p1_test, ROLLOUT_LEN,
                        )
                    )(rngs)

            t0 = time.time()
            info = jax.jit(jax.vmap(per_variant))(per_variant_rngs, flat_params)
            shaped = np.asarray(info["returned_episode_returns"])  # (V, NUM_EPS, n_agents)
            base = np.asarray(info["base_return"])
            shaped_pe = shaped.mean(axis=-1)  # (V, NUM_EPS)
            base_pe = base.mean(axis=-1)
            dt = time.time() - t0
            print(f"  [{position:9} bc={bc_run_id}] V={V} × NUM_EPS={NUM_EPS} = "
                  f"{V*NUM_EPS} rollouts in {dt:.1f}s "
                  f"(mean shaped: {shaped_pe.mean():.2f})", flush=True)

            for v in range(V):
                rid, tt, s = labels[v]
                results.append({
                    "run_id": rid, "teammate_set": tt, "position": position,
                    "bc_run_id": bc_run_id,
                    "seed_idx": s,
                    "shaped_per_ep": shaped_pe[v].tolist(),
                    "base_per_ep":   base_pe[v].tolist(),
                })

    return results, rng_master


def bootstrap_ci(values, n_resamples=10000, alpha=0.05, rng=None):
    """Percentile bootstrap CI of the mean."""
    if rng is None:
        rng = np.random.default_rng(0)
    values = np.asarray(values)
    n = len(values)
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = values[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return lo, hi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", default="coord_ring",
                        choices=sorted(LAYOUT_CFG.keys()))
    args = parser.parse_args()
    layout = args.layout

    if layout not in RUNS_BY_LAYOUT:
        raise SystemExit(f"No curated ego wandb runs for layout='{layout}'. "
                         f"Edit RUNS_BY_LAYOUT in this file to add them.")
    runs_for_layout = RUNS_BY_LAYOUT[layout]
    cfg = LAYOUT_CFG[layout]
    print(f"Layout: {layout}  do_reward_shaping={cfg['do_reward_shaping']}  "
          f"plot_metric={cfg['plot_metric']}  bc_selfplay_ref={cfg['bc_selfplay_ref']}",
          flush=True)
    print(f"Building env (kwargs={build_env_kwargs(layout)})...", flush=True)
    env = build_env(layout)
    print(f"Loading {len(BC_RUN_IDS)} BC policies (run_ids={list(BC_RUN_IDS)})...", flush=True)
    bc_policies = {rid: BCPolicy(layout, using_log_wrapper=True, run_id=rid)
                   for rid in BC_RUN_IDS}

    rng_master = jax.random.PRNGKey(EVAL_SEED)

    summary = {}
    for algo, run_specs in runs_for_layout.items():
        print(f"\n{'='*60}\nAlgo: {algo}\n{'='*60}", flush=True)
        rng_master, init_rng = jax.random.split(rng_master)
        ego_policy, _ = build_ego_policy(algo, env, init_rng)
        rows, rng_master = run_for_algo(layout, algo, env, ego_policy, bc_policies,
                                        run_specs, rng_master)
        summary[algo] = rows

    # Save raw per-episode JSON.
    out_dir = Path(f"results/overcooked-v1/{layout}")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "bc_xp_results.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved per-episode JSON -> {json_path}")

    # Per-row CSV (each row = one (algo, teammate_set, run_id, bc_run_id, position, seed_idx,
    # episode_idx) cell).
    per_ep_csv = out_dir / "bc_xp_results.csv"
    total_rows = 0
    with open(per_ep_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algo","teammate_set","run_id","bc_run_id","position","seed_idx",
                    "episode_idx","shaped_return","base_return"])
        for algo, rows in summary.items():
            for r in rows:
                for ep_idx, (sh, bs) in enumerate(zip(r["shaped_per_ep"], r["base_per_ep"])):
                    w.writerow([algo, r["teammate_set"], r["run_id"], r["bc_run_id"],
                                r["position"], r["seed_idx"], ep_idx,
                                round(sh, 4), round(bs, 4)])
                    total_rows += 1
    print(f"Saved per-episode CSV   -> {per_ep_csv} ({total_rows} rows)")

    # Aggregated CSV: mean ± std over (positions × bc_run_ids × seeds × episodes) per (algo, teammate_set).
    agg_csv = out_dir / "bc_xp_agg.csv"
    agg_ci_csv = out_dir / "bc_xp_agg_with_ci.csv"
    order_a = ["liam_ego", "ppo_ego", "meliba_ego"]
    # Iterate teammate sets in canonical order, but skip those with no runs in
    # this layout (e.g., cramped_room has no rotate runs).
    present_tts = {tt for specs in runs_for_layout.values() for _, tt in specs}
    order_t = [tt for tt in ("fcp", "rotate", "comedi") if tt in present_tts]
    plot_metric = cfg["plot_metric"]            # "shaped" | "base" — drives bar chart + bootstrap CI
    agg_table = []
    rng_boot = np.random.default_rng(20260504)

    print(f"\n{'algorithm':<11} {'teammate_set':<13} {'n':>5} "
          f"{'shaped (mean ± std)':<22} {'base (mean ± std)':<22} "
          f"{'CI95(' + plot_metric + ')':<22}")
    print("-" * 100)
    with open(agg_csv, "w", newline="") as f_agg, \
         open(agg_ci_csv, "w", newline="") as f_ci:
        w_agg = csv.writer(f_agg)
        w_ci = csv.writer(f_ci)
        w_agg.writerow(["algorithm","teammate_set","n_obs","shaped_mean","shaped_std",
                        "base_mean","base_std"])
        w_ci.writerow(["algorithm","teammate_set","n_obs","plot_metric",
                       "mean","std","se","ci_lo_boot","ci_hi_boot"])
        for tt in order_t:
            for algo in order_a:
                shaped = []
                base = []
                for r in summary[algo]:
                    if r["teammate_set"] != tt:
                        continue
                    shaped.extend(r["shaped_per_ep"])
                    base.extend(r["base_per_ep"])
                shaped = np.array(shaped); base = np.array(base)
                n = len(shaped)
                if n == 0:
                    continue
                mean_s, std_s = float(shaped.mean()), float(shaped.std())
                mean_b, std_b = float(base.mean()),   float(base.std())
                values = shaped if plot_metric == "shaped" else base
                mean_p, std_p = (mean_s, std_s) if plot_metric == "shaped" else (mean_b, std_b)
                se = std_p / np.sqrt(n) if n > 0 else 0.0
                ci_lo, ci_hi = bootstrap_ci(values, n_resamples=10000, rng=rng_boot)

                w_agg.writerow([algo, tt, n,
                                round(mean_s, 3), round(std_s, 3),
                                round(mean_b, 3), round(std_b, 3)])
                w_ci.writerow([algo, tt, n, plot_metric,
                               round(mean_p, 3), round(std_p, 3),
                               round(se, 3),
                               round(ci_lo, 3), round(ci_hi, 3)])
                agg_table.append({
                    "algorithm": algo, "teammate_set": tt, "n": n,
                    "shaped_mean": mean_s, "shaped_std": std_s,
                    "base_mean": mean_b, "base_std": std_b,
                    "plot_mean": mean_p, "plot_std": std_p,
                    "plot_se": se, "ci_lo": ci_lo, "ci_hi": ci_hi,
                })
                print(f"{algo:<11} {tt:<13} {n:>5} "
                      f"{mean_s:7.2f} ± {std_s:5.2f}      "
                      f"{mean_b:7.2f} ± {std_b:5.2f}      "
                      f"[{ci_lo:6.2f}, {ci_hi:6.2f}]")

    print(f"\nSaved aggregate CSV     -> {agg_csv}")
    print(f"Saved aggregate+CI CSV  -> {agg_ci_csv}")

    # ---- Bar chart: 2 groups (fcp, comedi); 3 bars per group (liam, ppo, meliba) ----
    bar_path = out_dir / "bc_xp_bar.png"
    plot_groups = [tt for tt in ("fcp", "comedi") if tt in present_tts]
    plot_algos = [("liam_ego", "LIAM", "tab:blue"),
                  ("ppo_ego",  "PPO",  "tab:green"),
                  ("meliba_ego","MELIBA","tab:red")]
    bc_ref = cfg["bc_selfplay_ref"]
    metric_label = "shaped return" if plot_metric == "shaped" else "base (sparse) return"

    # Lookup: (algo, teammate_set) -> row dict
    by_key = {(r["algorithm"], r["teammate_set"]): r for r in agg_table}

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    x = np.arange(len(plot_groups))
    bar_w = 0.25
    for i, (algo, label, color) in enumerate(plot_algos):
        means = [by_key[(algo, tt)]["plot_mean"] for tt in plot_groups]
        ci_lo = [by_key[(algo, tt)]["ci_lo"] for tt in plot_groups]
        ci_hi = [by_key[(algo, tt)]["ci_hi"] for tt in plot_groups]
        err_lo = [m - lo for m, lo in zip(means, ci_lo)]
        err_hi = [hi - m for m, hi in zip(means, ci_hi)]
        ax.bar(x + (i - 1) * bar_w, means, bar_w, label=label, color=color,
               yerr=[err_lo, err_hi], capsize=4, error_kw={"elinewidth": 1.2})

    ax.axhline(bc_ref, color="grey", linestyle="--", linewidth=1.2,
               label=f"BC×BC self-play ({bc_ref:.1f})")
    ax.set_xticks(x)
    ax.set_xticklabels([g.upper() for g in plot_groups])
    ax.set_ylabel(metric_label)
    ax.set_title(f"ego × BC (overcooked-v1/{layout})")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(bar_path, dpi=150)
    plt.close(fig)
    print(f"Saved bar chart         -> {bar_path}")


if __name__ == "__main__":
    main()
