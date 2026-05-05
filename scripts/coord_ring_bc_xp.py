"""Cross-play matrix: BC human-proxy × {ppo_ego, liam_ego, meliba_ego} on coord_ring.

Evaluates BOTH agent positions (ego at slot 0 vs BC at slot 0).

Vmap layout:
- Per (algo, position): one vmapped call over (run × seed) variants × episode rngs.
  All variants of the same algo share the actor architecture, so params are
  concatenated along the leading (seed) axis across runs and become a flat
  variant batch.

Usage:
  /scratch/cluster/jyliu/conda_envs/HANABI/bin/python scripts/coord_ring_bc_xp.py
"""
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
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

LAYOUT = "coord_ring"
NUM_EPS = 30
EVAL_SEED = 34957
ROLLOUT_LEN = 400

ENV_KWARGS = {
    "layout": LAYOUT,
    "random_obj_state": True,
    "do_reward_shaping": True,
    "reward_shaping_params": {
        "PLACEMENT_IN_POT_REW": 0.5,
        "PLATE_PICKUP_REWARD": 0.1,
        "SOUP_PICKUP_REWARD": 1.0,
        "ONION_PICKUP_REWARD": 0.1,
        "COUNTER_PICKUP_REWARD": 0,
        "COUNTER_DROP_REWARD": 0,
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

# Each run id is one wandb training run. Listed in fcp/rotate/comedi order.
RUNS = {
    "ppo_ego":    [("0wzr6nbv", "fcp"), ("7mrfwnra", "rotate"), ("5njknt2q", "comedi")],
    "liam_ego":   [("e8y4gy49", "fcp"), ("4vdq0mjv", "rotate"), ("pmm9u33p", "comedi")],
    "meliba_ego": [("x6gmp9uc", "fcp"), ("c9o32dg3", "rotate"),
                   ("bd6w70tr", "comedi"), ("7a19mtio", "comedi")],
}


def build_env():
    return LogWrapper(make_env("overcooked-v1", ENV_KWARGS))


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


def run_for_algo(algo, env, ego_policy, bc_policy, run_specs, rng_master):
    """Run all (run, seed, position, episode) for one algorithm in one vmapped pass per position.

    Args:
        run_specs: list of (run_id, teammate_set) for this algo.
    Returns:
        list of dicts: {run_id, teammate_set, position, seed_idx, shaped_per_ep, base_per_ep}
    """
    # Load all runs' params and stack to a flat variant batch.
    per_run_meta = []
    per_run_params = []
    for rid, tt in run_specs:
        path = f"results/overcooked-v1/{LAYOUT}/{algo}/wandb_{rid}"
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
        rng_master, rng_pos = jax.random.split(rng_master)
        # Per-variant rng → V×NUM_EPS rngs after inner split.
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
        # info leaves shape (V, NUM_EPS, ...).
        shaped = np.asarray(info["returned_episode_returns"])  # (V, NUM_EPS, n_agents)
        base = np.asarray(info["base_return"])
        # mean over agents (coop env: same return on both)
        shaped_pe = shaped.mean(axis=-1)  # (V, NUM_EPS)
        base_pe = base.mean(axis=-1)
        dt = time.time() - t0
        print(f"  [{position:9}] V={V} × NUM_EPS={NUM_EPS} = {V*NUM_EPS} rollouts in {dt:.1f}s "
              f"(mean shaped per-variant: {shaped_pe.mean(axis=1).mean():.2f})", flush=True)

        for v in range(V):
            rid, tt, s = labels[v]
            results.append({
                "run_id": rid, "teammate_set": tt, "position": position,
                "seed_idx": s,
                "shaped_per_ep": shaped_pe[v].tolist(),
                "base_per_ep":   base_pe[v].tolist(),
            })

    return results, rng_master


def main():
    print("Building env...", flush=True)
    env = build_env()
    print("Loading BC policy...", flush=True)
    bc_policy = BCPolicy(LAYOUT, using_log_wrapper=True)

    rng_master = jax.random.PRNGKey(EVAL_SEED)

    summary = {}
    for algo, run_specs in RUNS.items():
        print(f"\n{'='*60}\nAlgo: {algo}\n{'='*60}", flush=True)
        rng_master, init_rng = jax.random.split(rng_master)
        ego_policy, _ = build_ego_policy(algo, env, init_rng)
        rows, rng_master = run_for_algo(algo, env, ego_policy, bc_policy, run_specs, rng_master)
        summary[algo] = rows

    # Save raw per-episode JSON.
    out_dir = Path(f"results/overcooked-v1/{LAYOUT}")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "bc_xp_results.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved per-episode JSON -> {json_path}")

    # Per-row CSV (each row = one (algo, teammate_set, run_id, position, seed_idx, episode_idx) cell).
    import csv
    per_ep_csv = out_dir / "bc_xp_results.csv"
    with open(per_ep_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algo","teammate_set","run_id","position","seed_idx","episode_idx",
                    "shaped_return","base_return"])
        for algo, rows in summary.items():
            for r in rows:
                for ep_idx, (sh, bs) in enumerate(zip(r["shaped_per_ep"], r["base_per_ep"])):
                    w.writerow([algo, r["teammate_set"], r["run_id"], r["position"],
                                r["seed_idx"], ep_idx, round(sh, 4), round(bs, 4)])
    print(f"Saved per-episode CSV   -> {per_ep_csv}")

    # Aggregated CSV: mean ± std over (positions × seeds × episodes) per (algo, teammate_set).
    agg_csv = out_dir / "bc_xp_agg.csv"
    order_a = ["liam_ego", "ppo_ego", "meliba_ego"]
    order_t = ["fcp", "rotate", "comedi"]
    rows_for_agg = []
    print(f"\n{'algorithm':<11} {'teammate_set':<13} {'n':>5} {'shaped (mean ± std)':<22} {'base (mean ± std)':<22}")
    print("-" * 80)
    with open(agg_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algorithm","teammate_set","n_obs","shaped_mean","shaped_std",
                    "base_mean","base_std"])
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
                w.writerow([algo, tt, len(shaped),
                            round(float(shaped.mean()), 3), round(float(shaped.std()), 3),
                            round(float(base.mean()), 3), round(float(base.std()), 3)])
                print(f"{algo:<11} {tt:<13} {len(shaped):>5} "
                      f"{shaped.mean():7.2f} ± {shaped.std():5.2f}      "
                      f"{base.mean():7.2f} ± {base.std():5.2f}")
    print(f"\nSaved aggregate CSV     -> {agg_csv}")


if __name__ == "__main__":
    main()
