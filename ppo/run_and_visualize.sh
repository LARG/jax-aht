#!/bin/bash

# quick script for automating the process of training and visualizing the result for overcooked

# Set the base directory
result_dir="/scratch/cluster/jyliu/Documents/jax-aht/results/overcooked-v1/counter_circuit/ippo"

# Do the training
echo running ppo/run.py
python ppo/run.py -cn ippo_cctest

# Get the most recently modified subfolder
latest_result=$(ls -td "$result_dir"/*/ | head -n 1)

# generate the video
echo generating the video with $latest_result
python evaluation/vis_episodes.py $latest_result
