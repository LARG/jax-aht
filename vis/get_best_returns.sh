#!/bin/bash

# List all available tasks
python paper_vis/compute_best_returns.py list

# Compute best returns for all available tasks
python paper_vis/compute_best_returns.py compute

# Compute best returns for specific tasks
python paper_vis/compute_best_returns.py compute --tasks lbf overcooked-v1/cramped_room

# Force recomputation of best returns
python paper_vis/compute_best_returns.py compute --tasks lbf --force

# Show the computed best returns for specific tasks
python paper_vis/compute_best_returns.py show --tasks lbf overcooked-v1/cramped_room
