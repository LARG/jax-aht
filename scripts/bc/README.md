# LBF Human-Policy Experiments

This directory contains the behavior-cloning (BC) and human-regularized
self-play code used for the LBF human-policy experiments in the paper. It is
kept on this experiment branch rather than merged into the main `expts`
branch.

## Behavior cloning

Processed human trajectories are loaded with
`human_data_processing.load_lbf_data.load_bc_data_padded`; data preparation is
documented in [`human_data_processing/README.md`](../../human_data_processing/README.md).
`train.py` trains the recurrent BC policy. The optional `path` feature mode
uses `lbf_features.py` and is specific to this BC workflow.

```bash
python scripts/bc/train.py \
  --lbf_config <lbf-config> \
  --config agents/bc/configs/lbf.yaml \
  --lbf_feature_mode path \
  --output <checkpoint>.safetensors
```

The resulting checkpoint can be evaluated with `evaluate_lbf.py` or
`evaluate_lbf_dataset_partners.py`. Use `select_lbf_checkpoint_by_return.py`
to select checkpoints by rollout return rather than validation accuracy alone.

## Human-regularized self-play

`run_hr_ippo_lbf_sweep.py` runs parameter-sharing IPPO self-play from
`marl/ippo.py`. For nonzero regularization coefficients, the IPPO objective
adds a KL penalty against a frozen BC reference policy. A zero coefficient is
the unregularized IPPO baseline; `--include_bc` adds the BC policy to the same
output table.

```bash
python scripts/bc/run_hr_ippo_lbf_sweep.py \
  --lbf_config <lbf-config> \
  --human_reg_coef <coefficients...> \
  --human_ref_bc_checkpoint <checkpoint>.safetensors \
  --human_ref_bc_config <checkpoint>.yaml \
  --include_bc \
  --output_csv <results>.csv
```

The paper workflow used this self-play path. The separate PPO-ego/GAIL
experiment was not part of that workflow and is not included on this branch.
