#!/bin/bash
# A/B test: ppo_ego (S5 recurrent ego) on overcooked coord_ring vs comedi hparam train teammates.
# Usage: scripts_ab_ppo.sh <workdir> <gpu> <label> [total_timesteps]
WORKDIR=$1; GPU=$2; LABEL=$3; TT=${4:-1e7}
cd $WORKDIR || exit 1
export CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
PARTNER=/scratch/cluster/clw4542/explore_marl/jax-aht/hparam_train_teammates/hparam_search/overcooked-v1/coord_ring/comedi/hparam_search_train_teammates/2026-08-23_23-18-34/saved_train_run
exec timeout 28800 /scratch/cluster/clw4542/conda_envs/oeaht/bin/python ego_agent_training/run.py \
  algorithm=ppo_ego/overcooked-v1/coord_ring task=overcooked-v1/coord_ring \
  logger.mode=disabled run_heldout_eval=false label=$LABEL \
  algorithm.TOTAL_TIMESTEPS=$TT \
  algorithm.NUM_EGO_TRAIN_SEEDS=3 \
  algorithm.partner_agent.ippo.path=$PARTNER \
  algorithm.partner_agent.ippo.actor_type=actor_with_conditional_critic \
  algorithm.partner_agent.ippo.ckpt_key=final_params_conf \
  algorithm.partner_agent.ippo.idx_list=null \
  +algorithm.partner_agent.ippo.POP_SIZE=10
