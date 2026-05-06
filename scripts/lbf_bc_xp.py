"""Cross-play matrix: LBF BC-LSTM human-proxy × {ppo_ego, liam_ego, meliba_ego}
on overcooked-1 lbf/lbf_12x12.

Mirrors scripts/coord_ring_bc_xp.py but specialized to LBF:
- single BC checkpoint per layout (no `bc_run_id ∈ {0..4}` axis)
- LBF env (sparse reward only — no shaped/base distinction)
- avail_actions read off env_state directly (LBF wrapper exposes it; LogWrapper
  drops it for LBF, so we don't wrap)
- shorter rollouts: ROLLOUT_LEN=128

Vmap layout (per (algo, position)): one JIT call over (V variants × NUM_EPS
episodes) where V is the number of ego seeds for that algo (here 5+5=10 across
the two teammate sets).

Usage:
  /scratch/cluster/jyliu/conda_envs/HANABI/bin/python scripts/lbf_bc_xp.py [--layout lbf_12x12]
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

from agents.bc.lbf_human_proxy import load_lbf_human_proxy
from agents.initialize_agents import (
    initialize_liam_agent,
    initialize_meliba_agent,
    initialize_s5_agent,
)
from common.save_load_utils import load_train_run
from envs import make_env

NUM_EPS = 50
EVAL_SEED = 34957
ROLLOUT_LEN = 128
ACTION_DIM = 6

# LBF env construction kwargs per layout. The layout key matches the wandb
# TASK_NAME of the ego runs (e.g. lbf_7x7_nolevels, not lbf_7x7) — that's the
# env those egos were trained against. The HF BC was trained on grid7_food3_LEVELS
# but we evaluate against the egos' env (no levels) and let the wrapper auto-pad.
LAYOUT_KWARGS = {
    "lbf_12x12":         {"grid_size": 12, "num_food": 6, "different_levels": True},
    "lbf_7x7_nolevels":  {},  # default LBF: grid_size=7, num_food=3, different_levels=False
}

# Maps layout-key → HF BC name. The HF dataset has "lbf_7x7" (no _nolevels
# suffix) and "lbf_12x12"; both safetensors files are byte-identical anyway.
LAYOUT_TO_BC_HF = {
    "lbf_12x12":         "lbf_12x12",
    "lbf_7x7_nolevels":  "lbf_7x7",
}

# Per-layout BC×BC self-play reference (per-agent, greedy mode, matched env).
# Measured by running BCLSTMPolicyWrapper × BCLSTMPolicyWrapper for 64 episodes
# in the same env config the egos were trained on.
LAYOUT_CFG = {
    "lbf_12x12":         {"bc_selfplay_ref": 0.153, "bc_selfplay_label": "BC×BC self-play (greedy, ours)"},
    "lbf_7x7_nolevels":  {"bc_selfplay_ref": 0.279, "bc_selfplay_label": "BC×BC self-play (greedy, ours)"},
}

# Optional per-(layout, wandb_id) seed-allowlist. If present, ONLY the listed
# seed indices from that wandb run's `final_params` leading axis are kept; if
# absent, all seeds are used. Used to drop seeds that fully collapsed during
# training (e.g., 4/5 LIAM/fcp seeds on lbf_7x7_nolevels return 0.000 — the
# aggregate is dominated by those zeros, hiding the one healthy seed at 0.482).
SEED_ALLOWLIST = {
    "lbf_7x7_nolevels": {
        "zq6yx6kf": [1],         # liam_ego/fcp:    seeds 0,2,3,4 fully collapsed
        "0kh751xl": [1, 3, 4],   # liam_ego/comedi: seeds 0,2 fully collapsed
    },
}

# Curated wandb run ids per (layout, algo) → list of (wandb_id, teammate_set).
# Discovered manually via wandb API (filter neurips:benchmark + lbf/<task_name> +
# {fcp,comedi}_teammates tags). All cells have a single 5-seed run (no pooling).
RUNS_BY_LAYOUT = {
    "lbf_12x12": {
        "ppo_ego":    [("otxy993u", "fcp"), ("jna0irje", "comedi")],
        "liam_ego":   [("j9adbki8", "fcp"), ("3h6fcrni", "comedi")],
        "meliba_ego": [("pifmvl5o", "fcp"), ("pwvgckir", "comedi")],
    },
    "lbf_7x7_nolevels": {
        "ppo_ego":    [("d411alba", "fcp"), ("2e5x3uko", "comedi")],
        "liam_ego":   [("zq6yx6kf", "fcp"), ("0kh751xl", "comedi")],
        "meliba_ego": [("47gw238c", "fcp"), ("bc1legak", "comedi")],
    },
}


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
    # LBF LIAM was trained with 128-d encoder/decoder (overcooked uses 64-d).
    # Confirmed by inspecting the saved final_params shapes:
    #   ScannedLSTM input kernels = (30, 128), decoder Dense_0 = (20, 128).
    "ENCODER_HIDDEN_DIM": 128,
    "ENCODER_OUTPUT_DIM": 20,
    "DECODER_HIDDEN_DIM": 128,
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


def build_env(layout):
    return make_env("lbf", LAYOUT_KWARGS[layout])


def build_ego_policy(algo, env, init_rng):
    if algo == "ppo_ego":
        return initialize_s5_agent(PPO_EGO_CFG, env, init_rng)
    if algo == "liam_ego":
        cfg = dict(LIAM_CFG)
        cfg["POLICY_INPUT_DIM"] = (
            cfg["ENCODER_OUTPUT_DIM"] + env.observation_space(env.agents[0]).shape[0]
        )
        return initialize_liam_agent(cfg, env, init_rng)
    if algo == "meliba_ego":
        cfg = dict(MELIBA_CFG)
        cfg["POLICY_INPUT_DIM"] = (
            cfg["ENCODER_LATENT_DIM"] * 4 + env.observation_space(env.agents[0]).shape[0]
        )
        return initialize_meliba_agent(cfg, env, init_rng)
    raise ValueError(algo)


def extract_ego_params(algo, final):
    if algo == "ppo_ego":
        return final
    return {
        "encoder": final["encoder"],
        "decoder": final["decoder"],
        "policy":  final["policy"],
    }


def rollout_single_ep(rng, env, p0_pol, p0_param, p0_test_mode,
                      p1_pol, p1_param, p1_test_mode, max_steps):
    """One LBF episode. Per-agent return (cumulative reward over the trajectory)
    is computed manually since LogWrapper drops the LBF state's avail_actions.
    Both agents receive aux_obs=(prev_act_onehot, joint_act_onehot, prev_reward)
    so LIAM/MeLIBA encoders work; the BC ignores aux_obs."""
    rng, reset_rng = jax.random.split(rng)
    obs, env_state = env.reset(reset_rng)
    init_done = {k: jnp.zeros((), dtype=bool) for k in env.agents + ["__all__"]}
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
    obs, env_state, reward, done, _ = env.step(ks, env_state, env_act)

    init_carry = (
        1, env_state, obs, rng, done, reward, env_act_oh, h0, h1,
        # cumulative per-agent return (cooperative env -> agents tied; we use
        # mean of the two slots for symmetry with the overcooked script).
        (reward["agent_0"] + reward["agent_1"]) / 2.0,
        ~done["__all__"],   # alive flag
    )

    def scan_step(carry, _):
        def take(carry):
            ts, env_state, obs, rng, done, reward, act_oh, h0, h1, ep_ret, alive = carry
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
            obs_n, env_state_n, reward_n, done_n, _ = env.step(ks, env_state, env_act)
            r = (reward_n["agent_0"] + reward_n["agent_1"]) / 2.0
            ep_ret_n = ep_ret + r * alive
            alive_n = alive & ~done_n["__all__"]
            return (ts + 1, env_state_n, obs_n, rng, done_n, reward_n, env_act_oh,
                    h0n, h1n, ep_ret_n, alive_n)
        return jax.lax.cond(carry[4]["__all__"], lambda c: c, take, carry), None

    final, _ = jax.lax.scan(scan_step, init_carry, None, length=max_steps)
    return final[-2]   # ep_ret (per-agent cumulative)


def run_for_algo(layout, algo, env, ego_policy, bc_policy, bc_params, run_specs, rng_master):
    """Run all (run, seed, position, episode) for one algorithm.

    One vmapped pass per `position` — variant batch is V × NUM_EPS where V is
    the total number of ego seeds across the algo's runs.
    """
    per_run_meta = []
    per_run_params = []
    layout_allow = SEED_ALLOWLIST.get(layout, {})
    for rid, tt in run_specs:
        path = f"results/lbf/{layout}/{algo}/wandb_{rid}"
        out = load_train_run(path)
        params = extract_ego_params(algo, out["final_params"])
        n_seeds = jax.tree.leaves(params)[0].shape[0]
        seed_orig_idxs = list(range(n_seeds))
        if rid in layout_allow:
            keep = layout_allow[rid]
            keep_arr = jnp.asarray(keep)
            params = jax.tree.map(lambda v: v[keep_arr], params)
            seed_orig_idxs = list(keep)
            print(f"    [seed-filter] {rid}: keep {keep} of {n_seeds} → {len(keep)} variants",
                  flush=True)
            n_seeds = len(keep)
        per_run_meta.append((rid, tt, seed_orig_idxs))
        per_run_params.append(params)

    flat_params = jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *per_run_params)
    V = jax.tree.leaves(flat_params)[0].shape[0]
    print(f"  total variants V={V} (runs: {[(m[0], len(m[2])) for m in per_run_meta]})", flush=True)

    labels = []
    for rid, tt, seed_orig_idxs in per_run_meta:
        for s in seed_orig_idxs:
            labels.append((rid, tt, s))
    assert len(labels) == V

    results = []
    for position in ("ego_first", "bc_first"):
        rng_master, rng_pos = jax.random.split(rng_master)
        per_variant_rngs = jax.random.split(rng_pos, V)

        if position == "ego_first":
            p0_pol, p1_pol = ego_policy, bc_policy
            p0_test, p1_test = False, True

            def per_variant(rng, ego_p):
                rngs = jax.random.split(rng, NUM_EPS)
                return jax.vmap(lambda r: rollout_single_ep(
                    r, env, p0_pol, ego_p, p0_test,
                    p1_pol, bc_params, p1_test, ROLLOUT_LEN,
                ))(rngs)
        else:
            p0_pol, p1_pol = bc_policy, ego_policy
            p0_test, p1_test = True, False

            def per_variant(rng, ego_p):
                rngs = jax.random.split(rng, NUM_EPS)
                return jax.vmap(lambda r: rollout_single_ep(
                    r, env, p0_pol, bc_params, p0_test,
                    p1_pol, ego_p, p1_test, ROLLOUT_LEN,
                ))(rngs)

        t0 = time.time()
        rets = jax.jit(jax.vmap(per_variant))(per_variant_rngs, flat_params)
        rets = np.asarray(rets)   # shape (V, NUM_EPS)
        dt = time.time() - t0
        print(f"  [{position:9}] V={V} × NUM_EPS={NUM_EPS} = {V*NUM_EPS} rollouts in {dt:.1f}s "
              f"(mean: {rets.mean():.4f})", flush=True)

        for v in range(V):
            rid, tt, s = labels[v]
            results.append({
                "run_id": rid, "teammate_set": tt, "position": position,
                "seed_idx": s,
                "ret_per_ep": rets[v].tolist(),
            })

    return results, rng_master


def bootstrap_ci(values, n_resamples=10000, alpha=0.05, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    values = np.asarray(values)
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = values[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", default="lbf_12x12",
                        choices=sorted(RUNS_BY_LAYOUT.keys()))
    args = parser.parse_args()
    layout = args.layout

    runs_for_layout = RUNS_BY_LAYOUT[layout]
    cfg = LAYOUT_CFG[layout]
    print(f"Layout: {layout}  bc_selfplay_ref={cfg['bc_selfplay_ref']}", flush=True)
    print(f"Building env (kwargs={LAYOUT_KWARGS[layout]})...", flush=True)
    env = build_env(layout)

    bc_hf_name = LAYOUT_TO_BC_HF[layout]
    print(f"Loading LBF BC human-proxy ({bc_hf_name}) for {layout}...", flush=True)
    bc_policy, bc_params = load_lbf_human_proxy(bc_hf_name)
    rng_master = jax.random.PRNGKey(EVAL_SEED)

    summary = {}
    for algo, run_specs in runs_for_layout.items():
        print(f"\n{'='*60}\nAlgo: {algo}\n{'='*60}", flush=True)
        rng_master, init_rng = jax.random.split(rng_master)
        ego_policy, _ = build_ego_policy(algo, env, init_rng)
        rows, rng_master = run_for_algo(layout, algo, env, ego_policy, bc_policy, bc_params,
                                        run_specs, rng_master)
        summary[algo] = rows

    out_dir = Path(f"results/lbf/{layout}")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "bc_xp_results.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved per-episode JSON -> {json_path}")

    per_ep_csv = out_dir / "bc_xp_results.csv"
    total_rows = 0
    with open(per_ep_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algo", "teammate_set", "run_id", "position", "seed_idx",
                    "episode_idx", "return"])
        for algo, rows in summary.items():
            for r in rows:
                for ep_idx, v in enumerate(r["ret_per_ep"]):
                    w.writerow([algo, r["teammate_set"], r["run_id"],
                                r["position"], r["seed_idx"], ep_idx, round(v, 6)])
                    total_rows += 1
    print(f"Saved per-episode CSV   -> {per_ep_csv} ({total_rows} rows)")

    # Aggregate
    agg_csv    = out_dir / "bc_xp_agg.csv"
    agg_ci_csv = out_dir / "bc_xp_agg_with_ci.csv"
    order_a = ["liam_ego", "ppo_ego", "meliba_ego"]
    present_tts = {tt for specs in runs_for_layout.values() for _, tt in specs}
    order_t = [tt for tt in ("fcp", "rotate", "comedi") if tt in present_tts]
    rng_boot = np.random.default_rng(20260506)
    agg_table = []

    print(f"\n{'algorithm':<11} {'teammate_set':<13} {'n':>5} {'mean ± std':<22} {'CI95':<22}")
    print("-" * 80)
    with open(agg_csv, "w", newline="") as f_agg, \
         open(agg_ci_csv, "w", newline="") as f_ci:
        w_agg = csv.writer(f_agg)
        w_ci  = csv.writer(f_ci)
        w_agg.writerow(["algorithm", "teammate_set", "n_obs", "mean", "std"])
        w_ci.writerow(["algorithm", "teammate_set", "n_obs", "mean", "std",
                       "se", "ci_lo_boot", "ci_hi_boot"])
        for tt in order_t:
            for algo in order_a:
                vals = []
                for r in summary[algo]:
                    if r["teammate_set"] != tt:
                        continue
                    vals.extend(r["ret_per_ep"])
                vals = np.array(vals)
                n = len(vals)
                if n == 0:
                    continue
                mean, std = float(vals.mean()), float(vals.std())
                se = std / np.sqrt(n)
                lo, hi = bootstrap_ci(vals, n_resamples=10000, rng=rng_boot)
                w_agg.writerow([algo, tt, n, round(mean, 4), round(std, 4)])
                w_ci.writerow([algo, tt, n, round(mean, 4), round(std, 4),
                               round(se, 4), round(lo, 4), round(hi, 4)])
                agg_table.append({"algorithm": algo, "teammate_set": tt, "n": n,
                                  "mean": mean, "std": std, "ci_lo": lo, "ci_hi": hi})
                print(f"{algo:<11} {tt:<13} {n:>5} {mean:7.4f} ± {std:5.4f}      "
                      f"[{lo:6.4f}, {hi:6.4f}]")
    print(f"\nSaved aggregate CSV     -> {agg_csv}")
    print(f"Saved aggregate+CI CSV  -> {agg_ci_csv}")

    # Bar chart
    bar_path = out_dir / "bc_xp_bar.png"
    plot_groups = [tt for tt in ("fcp", "comedi") if tt in present_tts]
    plot_algos = [("liam_ego", "LIAM", "tab:blue"),
                  ("ppo_ego",  "PPO",  "tab:green"),
                  ("meliba_ego","MELIBA","tab:red")]
    bc_ref = cfg["bc_selfplay_ref"]
    by_key = {(r["algorithm"], r["teammate_set"]): r for r in agg_table}

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    x = np.arange(len(plot_groups))
    bar_w = 0.25
    for i, (algo, label, color) in enumerate(plot_algos):
        means = [by_key[(algo, tt)]["mean"] for tt in plot_groups]
        ci_lo = [by_key[(algo, tt)]["ci_lo"] for tt in plot_groups]
        ci_hi = [by_key[(algo, tt)]["ci_hi"] for tt in plot_groups]
        err_lo = [m - lo for m, lo in zip(means, ci_lo)]
        err_hi = [hi - m for m, hi in zip(means, ci_hi)]
        ax.bar(x + (i - 1) * bar_w, means, bar_w, label=label, color=color,
               yerr=[err_lo, err_hi], capsize=4, error_kw={"elinewidth": 1.2})
    ax.axhline(bc_ref, color="grey", linestyle="--", linewidth=1.2,
               label=f"{cfg['bc_selfplay_label']} ({bc_ref:.3f})")
    ax.set_xticks(x)
    ax.set_xticklabels([g.upper() for g in plot_groups])
    ax.set_ylabel("episode return (per-agent)")
    ax.set_title(f"ego × LBF-BC ({layout})")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(bar_path, dpi=150)
    plt.close(fig)
    print(f"Saved bar chart         -> {bar_path}")


if __name__ == "__main__":
    main()
