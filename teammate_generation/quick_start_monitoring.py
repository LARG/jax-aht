#!/usr/bin/env python
"""
Quick Start: BRDiv Monitoring

This script demonstrates the easiest way to run BRDiv with monitoring.
"""

# ============================================================================
# QUICK START GUIDE
# ============================================================================
#
# Step 1: Run BRDiv with monitoring enabled
# -----------------------------------------
# python teammate_generation/run.py \
#     algorithm=brdiv/lbf \
#     task=lbf \
#     label=my_brdiv_run \
#     enable_brdiv_monitoring=true \
#     brdiv_monitoring_dir=./my_results
#
# Step 2: Analyze the results
# ---------------------------------
# python teammate_generation/brdiv_monitoring_analysis.py \
#     ./my_results/brdiv_monitoring_data.json --detailed
#
# Output:
# ├── ./my_results/
# │   ├── brdiv_monitoring_data.json          (raw data)
# │   ├── brdiv_monitoring_plot.png           (basic plots)
# │   └── brdiv_monitoring_detailed_plot.png  (detailed plots, if --detailed)
#
# ============================================================================

import json
import subprocess
from pathlib import Path


def run_brdiv_with_monitoring(
    algorithm: str = "brdiv/lbf",
    task: str = "lbf",
    label: str = "brdiv_test",
    output_dir: str = "./brdiv_results",
    **kwargs
):
    """
    Run BRDiv with monitoring enabled.
    
    Parameters:
    -----------
    algorithm : str
        Algorithm config (e.g., "brdiv/lbf", "brdiv/overcooked")
    task : str
        Task config (e.g., "lbf", "overcooked")
    label : str
        Label for the experiment
    output_dir : str
        Directory to save monitoring output
    **kwargs : dict
        Additional arguments passed to run.py
    """
    
    cmd = [
        "python", "teammate_generation/run.py",
        f"algorithm={algorithm}",
        f"task={task}",
        f"label={label}",
        "enable_brdiv_monitoring=true",
        f"brdiv_monitoring_dir={output_dir}",
        "run_heldout_eval=false",
        "train_ego=false",
    ]
    
    # Add any extra arguments
    for key, value in kwargs.items():
        cmd.append(f"{key}={value}")
    
    print(f"Running: {' '.join(cmd)}")
    print(f"Output will be saved to: {output_dir}")
    print()
    
    result = subprocess.run(cmd)
    return result.returncode


def analyze_results(data_file: str, detailed: bool = True):
    """
    Analyze BRDiv monitoring results.
    
    Parameters:
    -----------
    data_file : str
        Path to brdiv_monitoring_data.json
    detailed : bool
        Whether to create detailed plots
    """
    
    cmd = ["python", "teammate_generation/brdiv_monitoring_analysis.py", data_file]
    
    if detailed:
        cmd.append("--detailed")
    
    print(f"Running: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd)
    return result.returncode


def main():
    """Example: Run BRDiv on LBF and analyze results."""
    
    # Step 1: Run BRDiv with monitoring
    print("="*70)
    print("Step 1: Running BRDiv with monitoring...")
    print("="*70)
    output_dir = "./brdiv_lbf_example"
    return_code = run_brdiv_with_monitoring(
        algorithm="brdiv/lbf",
        task="lbf",
        label="example_run",
        output_dir=output_dir,
    )
    
    if return_code != 0:
        print(f"BRDiv run failed with return code {return_code}")
        return
    
    # Step 2: Analyze results
    print()
    print("="*70)
    print("Step 2: Analyzing results...")
    print("="*70)
    data_file = f"{output_dir}/brdiv_monitoring_data.json"
    analyze_results(data_file, detailed=True)
    
    print()
    print("="*70)
    print("Complete! Check the output files:")
    print(f"  - Data: {data_file}")
    print(f"  - Plots: {output_dir}/brdiv_monitoring_*.png")
    print("="*70)


if __name__ == "__main__":
    main()
