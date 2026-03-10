# CREPPO on coop_recon (Continuous) — Session Runbook

> **Last updated:** 2026-03-09  
> **Purpose:** Hand-off reference for continuing CREPPO α-robustness work.  
> **Repo:** `jax-aht` (branch `social_laws`)

---

## 1. Background — PPO vs REPPO vs CREPPO

| Algorithm | Key idea | Where defined |
|---|---|---|
| **PPO** | Standard proximal policy optimisation | `ppo_single_agent_projection.py` |
| **REPPO** | REPPO adds a distributional (C51) Q-critic and a CVAE to model partner behaviour; uses epsilon-optimal action restriction at eval | `reppo_*` files |
| **CREPPO** | Conservative REPPO — clips the KL trust-region from below as well as above (the "conservative" constraint), preventing the adversary from deviating too far even during joint training | `creppo_*` files |

**JAX vs CREPPO:** JAX is the numerical backend (like NumPy + autograd + JIT). CREPPO is the RL algorithm *written in* JAX. The key reason to use JAX is `vmap`: we run **1,000 environments in parallel** across all CPU cores simultaneously, making training ~10–50× faster than sequential Python loops.

---

## 2. coop_recon_continuous Environment

### Task
Two agents on a 2D map split by an **opaque wall at x = 0.5**. Each agent must:
1. Detect water at their goal (80 % success rate, action 5)
2. Detect life at their goal (80 % success rate, action 6, requires water detected first)
3. Take a photo (100 % success, action 7, requires life detected)

Rewards: `+1.0` water, `+1.0` life, `+10.0` photo, `-0.01` per step.  
Theoretical max return per agent ≈ **11.45** (assuming ~15 steps to complete).

### Action space (8 total)
`0`=noop, `1-4`=move NSEW, `5`=detect_water, `6`=detect_life, `7`=take_photo

### Key environment config (`configs/task/continuous/coop_recon.yaml`)
```yaml
ENV_KWARGS:
  horizon: 40
  detection_radius: 0.2
  max_speed: 1.0
  movement_noise_std: 0.1   # 10% Gaussian noise on each step
  done_condition: "all"     # default; overridden in code
  ego_centric_obs: false
SINGLE_AGENT_1_PROJECTION: ""
SINGLE_AGENT_2_PROJECTION: ""
ALPHA_COST: false
```

### `done_condition` values
| Value | When episode ends |
|---|---|
| `"all"` | Both agents have taken their photos |
| `"any"` | Any agent takes a photo |
| `"agent_0"` | Agent 0 (focal) takes a photo — **used in worst-case joint eval** |
| `"agent_1"` | Agent 1 (focal) takes a photo — **used in worst-case joint eval** |

---

## 3. Alpha-Robustness

**Formula:** `α = worst_case_return[focal_agent] / optimal_return[focal_agent]`

- **optimal_return**: Focal agent plays its SAP policy; partner sends noop (frozen). Measured with `done_condition="any"` on `optimal_env`.
- **worst_case_return**: Both agents play the jointly-trained policy, restricted to epsilon-optimal actions from SAP Q-values. Measured with `done_condition=f"agent_{agent_idx}"` on `env` (so only the focal agent finishing can end the episode).
- **epsilon_optimal = 0.01** (absolute Q-value units, very tight — only 1-2 actions normally pass the mask).
- **Target:** α ≈ 1.0, meaning the adversary cannot reduce the focal agent's return (perfect spatial isolation).

---

## 4. TACC Hardware Comparison

| Partition | Cores | Memory BW | Notes |
|---|---|---|---|
| `skx-dev` | 48 × Skylake | DDR4 | 2-hour limit, good for quick tests |
| `skx` | 48 × Skylake | DDR4 | Unlimited time, but slower |
| `icx` | 80 × Ice Lake | DDR4+ | Better than SKX |
| `spr` ⭐ | 112 × Sapphire Rapids | HBM (3.5× BW) | **Best for JAX CPU training** |
| `gh` (H100) | 96 CPU + H100 GPU | HBM | Fastest but scarce; needs `jax[cuda12]` |

**Use `spr` for all CREPPO runs.** The HBM bandwidth on SPR eliminates the memory bottleneck that plagued SKX runs.

**Important:** JAX on TACC has a known `pthread_create` error if the PRNG subsystem can't spawn enough threads. You may see `F0306 Check failed: ret == 0 (11 vs. 0) Thread tf_XLAPjRtCpuClient` — this is a transient system issue, not a code bug.

---

## 5. SLURM Script (v16 — SAP domain randomization)

```bash
#!/bin/bash
#SBATCH -J jax_coop_recon_creppo
#SBATCH -o jax_coop_recon_creppo_%j.out
#SBATCH -e jax_coop_recon_creppo_%j.err
#SBATCH -p spr
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=112
#SBATCH -t 06:00:00
#SBATCH --mail-user=jeffreychen287@gmail.com
#SBATCH --mail-type=all

module load python
source /scratch/11079/jeffreychen287/jax-aht/venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD
export XLA_PYTHON_CLIENT_PREALLOCATE=false

/scratch/11079/jeffreychen287/jax-aht/venv/bin/python social_laws/run.py \
    task=continuous/coop_recon \
    algorithm=creppo/continuous/coop_recon \
    logger.mode=offline \
    algorithm.TOTAL_TIMESTEPS=20000000 \
    algorithm.FIXED_EVAL=true \
    algorithm.SAP_DOMAIN_RANDOMIZE_PARTNER=true \
    label=jax_coop_recon_v16_creppo

echo "Training complete."
```

**Submit:** `sbatch run_jax_continuous.slurm`  
**Monitor:** `squeue -u $USER`  
**Check outputs:** `tail jax_coop_recon_creppo_<JOBID>.{out,err}`

---

## 6. Syncing Results to W&B

```bash
# Find and sync all offline runs from a given version label
ls -td /scratch/11079/jeffreychen287/jax-aht/results/continuous/coop_recon/creppo_mlp/jax_coop_recon_v15_creppo/*/wandb/offline-run* \
  | xargs -I{} /scratch/11079/jeffreychen287/jax-aht/venv/bin/wandb sync {} \
    --entity jeffreychen287-the-university-of-texas-at-austin \
    --project aht-benchmark
```

Each completed run produces **4 offline W&B runs** (SAP Agent 1, SAP Agent 2, Joint Agent 1, Joint Agent 2). Expect them all in the `offline-run*` glob.

**W&B entity:** `jeffreychen287-the-university-of-texas-at-austin`  
**W&B project:** `aht-benchmark`

---

## 7. Run History & What Each Version Changed

### v13 — First full CREPPO run (10M steps, SPR)
- **Label:** `jax_coop_recon_v13_creppo`
- **Bugs fixed to get here:**
  1. `ConfigKeyError: done_condition` — OmegaConf struct didn't have `done_condition` declared. Fixed by adding `done_condition: "all"` to `coop_recon.yaml`.
  2. `optimal_env` in `creppo_joint.py` and `reppo_joint.py` was missing `done_condition="any"`, causing alpha ≈ 0.5 (episodes ran to horizon unnecessarily).
- **Result:** Alpha Agent 1 ≈ 0.58, Alpha Agent 2 ≈ 0.77
- **Problem:** Agent 1's SAP return was only ~5–6 (not converged). 10M steps insufficient.

### v14 — 20M steps + wrong done_condition fix
- **Label:** `jax_coop_recon_v14_creppo`
- **Changes:** TOTAL_TIMESTEPS→20M; added `done_condition="any"` to the worst-case `env` (in addition to `optimal_env`).
- **Result:** Agent 1 SAP converged to ~10 ✅. But Agent 2 alpha DROPPED from 0.77 → 0.39 ❌
- **Root cause of regression:** With `done_condition="any"` on the worst-case env, the adversary could complete its own task (in ~10 steps) and terminate the episode early before the focal agent finished, giving the focal agent only step penalties.

### v15 — Focal-agent-specific done condition (current best)
- **Label:** `jax_coop_recon_v15_creppo`
- **Changes:**
  1. Added `done_condition="agent_0"` and `"agent_1"` modes to `coop_recon_continuous_wrapper.py`
  2. Worst-case `env` now uses `done_condition=f"agent_{agent_idx}"` — episode ends ONLY when the focal agent takes its photo, preventing the adversary from exploiting early termination.
  3. 20M timesteps retained.
- **Commits:** `ab6577b` (done_condition fix, 3 files)
- **Result:** Alpha Agent 1 ≈ 0.67, Alpha Agent 2 ≈ 0.79. Both SAPs converged (~10–11 return each). ✅

---

## 8. Bug History & Fixes

| Bug | Symptom | Root Cause | Fix |
|---|---|---|---|
| Alpha ≈ 0.5 | `returned_episode_returns` too low | `optimal_env` missing `done_condition="any"` | Added override in `creppo_joint.py` & `reppo_joint.py` |
| `ConfigKeyError: done_condition` | Job crash at joint training start | OmegaConf struct didn't have the key declared | Added `done_condition: "all"` to `coop_recon.yaml` |
| Alpha regression 0.77→0.39 (v14) | Agent 2's worst-case return collapsed | `done_condition="any"` on worst-case env let adversary terminate episode by finishing its OWN task | New `"agent_i"` done condition in wrapper |
| Agent 1 SAP not converged (v13) | Alpha Agent 1 stuck at 0.58; SAP return ≈5–6 | 10M timesteps insufficient for Agent 0's SAP to converge | Increased to 20M timesteps |

---

## 9. Alpha Debugging Story (Current Open Issue)

### Current alpha: v15
- Agent 1 Optimize: α ≈ 0.67 (SAP optimal ≈12.3, worst-case ≈8.2)
- Agent 2 Optimize: α ≈ 0.79 (SAP optimal ≈11.85, worst-case ≈9.39)

### Why alpha ≠ 1.0?
`epsilon_optimal = 0.01` is extremely tight. The adversary is effectively restricted to playing its SAP-optimal strategy — nothing truly adversarial is happening. The gap comes from the **observation distribution shift**:

- **SAP training**: partner sends noop → `picture_taken[partner]` is always `False`, partner position is fixed at its initial position
- **Worst-case eval**: partner plays near-SAP-optimal → partner MOVES and eventually completes its own task (flipping `picture_taken[partner]` to `True`)
- **Result**: the focal agent sees out-of-distribution observations it was never trained on → suboptimal actions → lower return

### PI's assessment (2026-03-09)
> "Nothing truly adversarial is happening right now. The agents are still restricted by their optimal policies so they can't deviate from that. Actually, that is interesting. The grid env also never sees the goal completion of the other agent during the SAP. We should discuss tomorrow and whether we should induce the goal state for the other agents as part of the SAP. It should be easy to set that as an extra condition during randomization."

### Proposed fix: Domain Randomization in SAP
Randomize the partner's task state (`detected_water`, `detected_life`, `picture_taken`) and starting position at the beginning of each SAP episode (even though the partner still sends noop). This forces the focal agent's policy to become robust to whatever the partner's state might be.

**Files to change:**
- `coop_recon_continuous_wrapper.py` → `reset()` method: randomize partner task states
- `creppo_single_agent_projection.py` → pass a config flag to enable this during SAP training

**Discussed with PI on 2026-03-10. Implemented — see v16 below.**

---

### v16 — SAP Domain Randomization (alpha gap fix)
- **Label:** `jax_coop_recon_v16_creppo`
- **Date:** 2026-03-09
- **Root cause identified:** During SAP training, partner sends noop → `detected_water/life/picture_taken[partner]` are always `[False, False, False]`. During worst-case eval, partner plays near-optimally → those flags flip to `[True, True, True]` mid-episode. The focal agent's Q-network has never seen these states → out-of-distribution input → suboptimal actions → α gap.
- **Why PPO got α ≈ 1.0 but CREPPO didn't:** PPO uses a *separate* DQN value function (trained in the joint phase with both agents acting, so it sees partner task completion naturally). CREPPO reuses the SAP Q-values for ε-optimal masking — those Q-values are trained with noop partner and are OOD for partner-active states.
- **Fix:** SAP domain randomization — at each SAP episode `reset()`, the partner's task state is randomly initialized (progression-consistent: water → life → photo, each conditioned on the previous). Partner still sends noop during the episode; only the *initial observation distribution* widens.
- **`done_condition` change (SAP only):** With randomization, `done_condition="any"` would terminate episodes immediately (if partner starts with `picture_taken=True`, `jnp.any()` is already True after step 1). Changed to `done_condition=f"agent_{agent_idx}"` when randomization is ON.
- **Files changed:**
  - `envs/coop_recon_continuous/coop_recon_continuous_wrapper.py` — new `sap_domain_randomize_partner` + `sap_focal_agent_idx` kwargs, progressive randomization in `reset()`
  - `social_laws/creppo_single_agent_projection.py` — gates `done_condition` and new env kwargs on `SAP_DOMAIN_RANDOMIZE_PARTNER`
  - `social_laws/configs/algorithm/creppo/_base_.yaml` — new `SAP_DOMAIN_RANDOMIZE_PARTNER: false` flag
- **TACC command:**
  ```bash
  cd /scratch/11079/jeffreychen287/jax-aht && \
  PYTHONPATH=$PWD \
  /scratch/11079/jeffreychen287/jax-aht/venv/bin/python social_laws/run.py \
    task=continuous/coop_recon \
    algorithm=creppo/continuous/coop_recon \
    logger.mode=offline \
    algorithm.TOTAL_TIMESTEPS=20000000 \
    algorithm.FIXED_EVAL=true \
    algorithm.SAP_DOMAIN_RANDOMIZE_PARTNER=true \
    label=jax_coop_recon_v16_creppo
  ```
- **Expected improvement:** α → 0.85–1.0 (vs v15: α ≈ 0.67–0.79)
- **Revert:** Run without `algorithm.SAP_DOMAIN_RANDOMIZE_PARTNER=true` (or set to `false`). No existing code paths are affected unless the flag is explicitly enabled.

---

## 10. Key File Map

| File | Purpose |
|---|---|
| `social_laws/run.py` | Top-level orchestrator: runs SAP then joint for each agent |
| `social_laws/creppo_single_agent_projection.py` | SAP training (single agent, partner = noop) |
| `social_laws/creppo_joint.py` | Joint adversarial training + alpha evaluation |
| `social_laws/common/run_episodes_creppo_w_robustness.py` | Eval episode runner (worst-case + optimal scans) |
| `envs/coop_recon_continuous/coop_recon_continuous_wrapper.py` | The JAX environment |
| `envs/log_wrapper.py` | Accumulates `returned_episode_returns` per-agent |
| `social_laws/configs/task/continuous/coop_recon.yaml` | Task config (ENV_KWARGS, SAP instances, ALPHA_COST) |
| `social_laws/configs/algorithm/creppo/_base_.yaml` | Algorithm hyperparams incl. EPSILON_OPTIMAL, TOTAL_TIMESTEPS |
| `run_jax_continuous.slurm` | SLURM job script (at repo root) |

---

## 11. Useful TACC Commands

```bash
# Check job queue
squeue -u $USER

# Check node availability
sinfo -p spr

# Check disk quotas (watch /home1 < 14GB limit!)
quota -s

# Clean pip cache (safe, often frees GBs from /home1)
rm -rf ~/.cache/pip

# Check SU balance
sacct -u $USER --format=JobID,JobName,Partition,Elapsed,CPUTime,State

# Tail job output live
tail -F jax_coop_recon_creppo_<JOBID>.out
```

**SU balance as of 2026-03-09:** ~1901 SUs remaining (TG-CIS251073, expires 2026-10-14).  
Each SPR node-hour ≈ 2 SUs. ~950 SPR node-hours remain.

---

## 12. Environment Setup (if re-creating venv)

```bash
# Already set up at:
/scratch/11079/jeffreychen287/jax-aht/venv

# To activate
module load python
source /scratch/11079/jeffreychen287/jax-aht/venv/bin/activate

# Installed packages include: jax[cpu], flax, optax, gymnasium, hydra, wandb
# GPU JAX (for H100 runs, if needed):
# pip install -U "jax[cuda12]"
```
