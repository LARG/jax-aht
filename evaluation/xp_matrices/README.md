# Heldout cross-play matrices

Each `<task>.csv` holds the cross-play returns between every heldout teammate (`row_agent`, from
`evaluation/configs/global_heldout_settings.yaml`) and every best response (`col_agent`, from
`evaluation/configs/global_heldout_br.yaml`). `is_diag` marks the best response trained for that
row's teammate. The upper performance bounds in `global_heldout_settings.yaml` are at least
the largest value in each teammate's row.

All matrices use the heldout evaluation settings: `EVAL_SEED` 34957, 64 episodes per pair, and the
mean over episodes with a bootstrapped 95% CI. Returns are raw (not normalized), one row per
metric. The best response plays as `agent_0` and the teammate as `agent_1`, the same seating as
heldout evaluation (`seating = new`). `evaluation/run.py -cn heldout_xp` seats them the other way
round, so its numbers can differ slightly from these.

`hanabi_diag_flags.csv` lists Hanabi teammates whose diagonal entry may understate the true best
response, with the reason for each.
