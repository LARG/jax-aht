#!/bin/bash
# Train a best-response ego agent for each LBF heldout evaluation agent.
# Run from the repo root: bash evaluation/train_br_for_lbf_heldout.sh

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/ego_agent_training:$PYTHONPATH"

# Common overrides
COMMON="task=lbf algorithm=ppo_br/lbf run_heldout_eval=false algorithm.TOTAL_TIMESTEPS=1e6"

# ── RL-based heldout agents ──────────────────────────────────────────────────

# ippo_mlp (seed 0, final checkpoint)
# python ego_agent_training/run.py $COMMON \
#   label=br_ippo_mlp \
#   '~algorithm.partner_agent' '+algorithm.partner_agent={ippo_mlp: {path: "eval_teammates/lbf/ippo/2025-04-21_23-41-17/saved_train_run", actor_type: mlp, ckpt_key: final_params, idx_list: [0], test_mode: false}}'

# ippo_mlp_s2c0 (seed 2, final checkpoint)
python ego_agent_training/run.py $COMMON \
  label=br_ippo_mlp_s2c0 \
  '~algorithm.partner_agent' '+algorithm.partner_agent={ippo_mlp_s2c0: {path: "eval_teammates/lbf/ippo/2025-04-21_23-41-17/saved_train_run", actor_type: mlp, ckpt_key: final_params, idx_list: [2], test_mode: false}}'

# brdiv-conf1 member 0
python ego_agent_training/run.py $COMMON \
  label=br_brdiv_conf1_m0 \
  '~algorithm.partner_agent' '+algorithm.partner_agent={brdiv_conf1_m0: {path: "eval_teammates/lbf/brdiv/2025-04-16/11-32-07/saved_train_run", actor_type: actor_with_conditional_critic, ckpt_key: final_params_conf, idx_list: [0], POP_SIZE: 5, test_mode: false}}'

# brdiv-conf1 member 1
python ego_agent_training/run.py $COMMON \
  label=br_brdiv_conf1_m1 \
  '~algorithm.partner_agent' '+algorithm.partner_agent={brdiv_conf1_m1: {path: "eval_teammates/lbf/brdiv/2025-04-16/11-32-07/saved_train_run", actor_type: actor_with_conditional_critic, ckpt_key: final_params_conf, idx_list: [1], POP_SIZE: 5, test_mode: false}}'

# brdiv-conf1 member 2
python ego_agent_training/run.py $COMMON \
  label=br_brdiv_conf1_m2 \
  '~algorithm.partner_agent' '+algorithm.partner_agent={brdiv_conf1_m2: {path: "eval_teammates/lbf/brdiv/2025-04-16/11-32-07/saved_train_run", actor_type: actor_with_conditional_critic, ckpt_key: final_params_conf, idx_list: [2], POP_SIZE: 5, test_mode: false}}'

# brdiv-conf2 member 0
python ego_agent_training/run.py $COMMON \
  label=br_brdiv_conf2_m0 \
  '~algorithm.partner_agent' '+algorithm.partner_agent={brdiv_conf2_m0: {path: "eval_teammates/lbf/brdiv/2025-04-23/13-48-47/saved_train_run", actor_type: actor_with_conditional_critic, ckpt_key: final_params_conf, idx_list: [0], POP_SIZE: 3, test_mode: false}}'

# brdiv-conf2 member 1
python ego_agent_training/run.py $COMMON \
  label=br_brdiv_conf2_m1 \
  '~algorithm.partner_agent' '+algorithm.partner_agent={brdiv_conf2_m1: {path: "eval_teammates/lbf/brdiv/2025-04-23/13-48-47/saved_train_run", actor_type: actor_with_conditional_critic, ckpt_key: final_params_conf, idx_list: [1], POP_SIZE: 3, test_mode: false}}'

# ── Heuristic heldout agents ────────────────────────────────────────────────

python ego_agent_training/run.py $COMMON \
  label=br_seq_agent_lexi \
  '~algorithm.partner_agent' '+algorithm.partner_agent={seq_agent_lexi: {actor_type: seq_agent, ordering_strategy: lexicographic}}'

python ego_agent_training/run.py $COMMON \
  label=br_seq_agent_rlexi \
  '~algorithm.partner_agent' '+algorithm.partner_agent={seq_agent_rlexi: {actor_type: seq_agent, ordering_strategy: reverse_lexicographic}}'

python ego_agent_training/run.py $COMMON \
  label=br_seq_agent_col \
  '~algorithm.partner_agent' '+algorithm.partner_agent={seq_agent_col: {actor_type: seq_agent, ordering_strategy: column_major}}'

python ego_agent_training/run.py $COMMON \
  label=br_seq_agent_rcol \
  '~algorithm.partner_agent' '+algorithm.partner_agent={seq_agent_rcol: {actor_type: seq_agent, ordering_strategy: reverse_column_major}}'

python ego_agent_training/run.py $COMMON \
  label=br_seq_agent_nearest \
  '~algorithm.partner_agent' '+algorithm.partner_agent={seq_agent_nearest: {actor_type: seq_agent, ordering_strategy: nearest_agent}}'

python ego_agent_training/run.py $COMMON \
  label=br_seq_agent_farthest \
  '~algorithm.partner_agent' '+algorithm.partner_agent={seq_agent_farthest: {actor_type: seq_agent, ordering_strategy: farthest_agent}}'

echo "All BR training runs completed."
