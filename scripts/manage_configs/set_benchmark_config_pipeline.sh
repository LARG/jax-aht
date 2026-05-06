# Before running this pipeline, ensure that the best hparam wandb sweep ids are up to date 
# in scripts/paper_vis/plot_globals.py
# Remove dry run flags after verifying output!

# 1. For each task, fetch best hparam from sweep and apply them to each task/algo config.
# Use --force-recompute flag to force recomputation of best hparams 
# (instead of using locally cached values from wandb)
# python scripts/manage_configs/apply_best_hparams.py \
#     --task "lbf/lbf_7x7_nolevels" --dry-run

# python scripts/manage_configs/apply_best_hparams.py \
#     --task "mini-hanabi" --dry-run

# 2. Set benchmark timesteps
python scripts/manage_configs/update_timesteps.py teammate_generation/ \
    --easy-target 195M --hard-target 390M --dry-run

python scripts/manage_configs/update_timesteps.py open_ended_training/ \
    --easy-target 195M --hard-target 390M --skip-algos open_ended_minimax paired --dry-run

python scripts/manage_configs/update_timesteps.py ego_agent_training/ \
    --easy-target 30M --hard-target 60M --skip-algos ppo_br --dry-run

