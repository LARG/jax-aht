# Hanabi Benchmark

Cooperative card game benchmark for Ad Hoc Teamwork (AHT). Wraps
[JaxMARL](https://github.com/FLAIROx/JaxMARL)'s Hanabi environment
and adds partners, configs, and held-out eval across every pipeline
in the repo.

## Configurations

| Variant | Colors | Ranks | Hand | Info | Life | Obs dim | Actions | Max score |
|---------|:------:|:-----:|:----:|:----:|:----:|:-------:|:-------:|:---------:|
| Full Hanabi | 5 | 5 | 5 | 8 | 3 | 658 | 21 | 25 |
| Mini-Hanabi | 3 | 3 | 3 | 5 | 3 | 191 | 13 | 9  |

Task configs: `{pipeline}/configs/task/{hanabi,mini-hanabi}.yaml` in
every pipeline (`marl`, `ego_agent_training`, `teammate_generation`,
`open_ended_training`, `evaluation`).

## Agents

All agents live under `agents/hanabi/` and are pure JAX. Self-play
scores reported on full Hanabi.

### Rule-based

| Agent | File | Self-play/25 | Notes |
|-------|------|:------------:|-------|
| Random | `random_agent.py` | 0.0 | Uniform random legal action |
| RuleBased x4 | `rule_based_agent.py` | ~0 | Priority weights over action categories: `cautious`, `aggressive`, `communicative`, `frugal` |
| IGGI | `iggi_agent.py` | 11.7 | Play safe > hint playable > Osawa discard > discard oldest (Walton-Rivers 2017) |
| Piers | `piers_agent.py` | 11.6 | IGGI + probabilistic play (>60%) + dispensable hints |
| Van Den Bergh | `van_den_bergh_agent.py` | 9.3 | Probabilistic discard ordering |
| Outer | `outer_agent.py` | 0.0 | Hint-heavy (degenerate on full Hanabi) |
| Flawed | `flawed_agent.py` | 0.2 | IGGI with tunable mistake probability |
| SmartBot | `smartbot_*.py` | 18.0 | Convention-heavy, first JAX port (ref C++ 23.1) |

### Learned (pretrained)

| Agent | File | Self-play/25 | Notes |
|-------|------|:------------:|-------|
| OBL R2D2 L1-L5 | `obl_r2d2_agent.py` | 21-24 | Off-Belief Learning (Hu et al. 2021). Download via `bash agents/hanabi/download_obl_r2d2.sh` |
| BC-LSTM | `bc_lstm_agent.py` | 3.4 | Behavioral cloning on AH2AC2 dataset. 45.5% accuracy |

### Learned (trainable)

| Pipeline | Method | File |
|----------|--------|------|
| MARL self-play | IPPO | `marl/ippo.py` (existing) |
| MARL self-play | IPPO + Other-Play | `marl/ippo.py` + `agents/hanabi/other_play.py` (color permutation) |
| Ego training | PPO ego | `ego_agent_training/ppo_ego.py` (existing) |
| Ego training | LIAM ego | `ego_agent_training/liam_ego.py` (existing) |
| Ego training | MeLIBA ego | `ego_agent_training/meliba_ego.py` (existing) |
| Ego training | PPO-BR | `ego_agent_training/ppo_br.py` (existing) |
| Teammate gen | FCP | `teammate_generation/fcp.py` (existing) |
| Teammate gen | BRDiv | `teammate_generation/BRDiv.py` (existing) |
| Teammate gen | CoMeDi | `teammate_generation/CoMeDi.py` (existing) |
| Teammate gen | LBRDiv | `teammate_generation/LBRDiv.py` (existing) |
| Open-ended | ROTATE | `open_ended_training/rotate.py` (existing) |
| Open-ended | PAIRED | `open_ended_training/paired.py` (existing) |
| Open-ended | CoLE | `open_ended_training/cole.py` (existing) |
| Open-ended | Open-Ended Minimax | `open_ended_training/open_ended_minimax.py` (existing) |

## Held-out evaluation

Partners registered in `evaluation/configs/global_heldout_settings.yaml`
under `hanabi:` and `mini-hanabi:`. Wrapper dispatch is in
`evaluation/heldout_evaluator.py`.

| Partner | Full | Mini | Notes |
|---------|:----:|:----:|-------|
| `random_agent` | ✓ | ✓ | 0/25 floor |
| `rule_based_*` x4 | ✓ | ✓ | cautious, aggressive, communicative, frugal |
| `iggi` | ✓ | ✓ | |
| `piers` | ✓ | ✓ | |
| `flawed_30` / `flawed_60` | ✓ | ✓ | 30%, 60% mistake rate |
| `outer` | ✓ | ✓ | |
| `van_den_bergh` | ✓ | ✓ | |
| `smartbot` | ✓ | ✓ | card_counts threaded from env_kwargs for mini |
| `obl_r2d2` | ✓ | - | Full only; see limits note below |
| `bc_lstm_human_proxy` | ✓ | - | Full only; see limits note below |
| `ippo_seed123` | - | ✓ | Mini only (held-out IPPO partner) |

Full Hanabi: **14 partners**. Mini: **13 partners** (same minus
OBL and BC-LSTM).

**Why OBL/BC-LSTM don't run on mini Hanabi.** OBL's pretrained weights from
`mttga/obl-r2d2-flax` hardcode a 21-action output head and split the
obs at index 125. Neither lines up with mini's 13 actions or 191-dim
obs. BC-LSTM is trained on the AH2AC2 human play dataset, which is full-Hanabi only
(no mini human data to clone from). 

## Running

```bash
# MARL self-play
python marl/run.py task=hanabi algorithm=ippo/hanabi
python marl/run.py task=mini-hanabi algorithm=ippo/mini-hanabi
python marl/run.py task=hanabi algorithm=ippo/hanabi-otherplay     # Other-Play variant

# Ego training (requires held-out partners, see
# scripts/regenerate_hanabi_eval_teammates.sh)
python ego_agent_training/run.py task=hanabi algorithm=ppo_ego/hanabi
python ego_agent_training/run.py task=hanabi algorithm=liam_ego/hanabi

# Teammate generation + ego + held-out evaluation
python teammate_generation/run.py task=hanabi algorithm=fcp/hanabi \
    run_heldout_eval=true

# Smoke test all methods on mini-Hanabi (2 min total)
bash scripts/smoke_test_mini_hanabi.sh

# Agent self-play verification
python tests/test_hanabi_agents.py

# Cross-play matrix
bash scripts/run_hanabi_crossplay.sh
```

## Key findings

- IPPO self-play reaches 15.78/25 at 10^9 steps (3 seeds, 128 envs),
  in line with JaxMARL's scaling.
- LIAM ego beats PPO ego on held-out partners (3.32 vs 2.17 at 10^8
  steps, 15-partner FCP population). LIAM trains lower (6.39 vs 7.12)
  but transfers better - the encoder helps.
- Self-play strength trades off against coordination. SmartBot
  17.42 self-play drops to 0.40 vs the BC-LSTM human proxy. OBL-L4
  24.28 drops to 1.02. IGGI 11.64 drops to only 3.30. 

## References

- Bard et al. 2020. "The Hanabi Challenge: A New Frontier for AI Research"
- Walton-Rivers et al. 2017. "Evaluating and Modelling Hanabi-Playing Agents"
- Hu et al. 2020. "Other-Play for Zero-Shot Coordination"
- Hu et al. 2021. "Off-Belief Learning"
- Papoudakis et al. 2021. "Agent Modelling under Partial Observability"
- AH2AC2 2025. "Ad Hoc Human-Agent Coordination Challenge"
