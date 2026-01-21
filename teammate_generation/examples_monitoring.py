"""
Example usage of BRDiv monitoring system.

This file shows various ways to use the monitoring system.
"""

# =============================================================================
# EXAMPLE 1: Basic Usage via Command Line
# =============================================================================
# 
# Run BRDiv on LBF with monitoring:
#
# $ python teammate_generation/run.py \
#     algorithm=brdiv/lbf \
#     task=lbf \
#     label=exp1 \
#     enable_brdiv_monitoring=true \
#     brdiv_monitoring_dir=./exp1_results \
#     run_heldout_eval=false \
#     train_ego=false
#
# Output:
# $ ls ./exp1_results/
# brdiv_monitoring_data.json
# brdiv_monitoring_plot.png
#
# =============================================================================


# =============================================================================
# EXAMPLE 2: Analyze Results
# =============================================================================
#
# Analyze the monitoring data:
#
# $ python teammate_generation/brdiv_monitoring_analysis.py \
#     ./exp1_results/brdiv_monitoring_data.json --detailed
#
# Output: Console summary + brdiv_monitoring_detailed_plot.png
#
# =============================================================================


# =============================================================================
# EXAMPLE 3: Python - Basic Monitoring
# =============================================================================

def example_basic_monitoring():
    """Run BRDiv with monitoring via Python."""
    from teammate_generation.BRDiv import run_brdiv
    from common.wandb_visualizations import Logger
    
    # Create config
    config = {
        "algorithm": {
            "ENV_NAME": "lbf",
            "ENV_KWARGS": {},
            "TRAIN_SEED": 0,
            "NUM_SEEDS": 1,
            "PARTNER_POP_SIZE": 2,
            "NUM_ENVS": 64,
            "ROLLOUT_LENGTH": 128,
            "TOTAL_TIMESTEPS": 10000000,
            # ... other config params ...
        },
        "enable_brdiv_monitoring": True,
        "brdiv_monitoring_dir": "./my_results",
    }
    
    logger = Logger(config)
    
    # Run with monitoring (automatic)
    partner_params, partner_population = run_brdiv(config, logger)
    
    print("Results saved to ./my_results/")
    print("  - brdiv_monitoring_data.json")
    print("  - brdiv_monitoring_plot.png")


# =============================================================================
# EXAMPLE 4: Python - Manual Monitoring
# =============================================================================

def example_manual_monitoring():
    """Manual monitoring without run_brdiv integration."""
    from teammate_generation.brdiv_with_monitoring import BRDivMonitor
    import time
    
    # Create monitor
    monitor = BRDivMonitor(output_dir="./manual_monitoring")
    monitor.start()
    
    # Simulate training loop
    for update_step in range(10):
        time.sleep(1)  # Simulate training
        
        # Simulate returns
        sp_return = 0.4 + 0.02 * update_step
        xp_return = 0.2 + 0.03 * update_step
        
        # Record
        monitor.record_update(update_step, sp_return, xp_return)
    
    # Save and plot
    monitor.save_data()
    monitor.plot_results()
    
    print("Manual monitoring complete!")
    print("  - Data: ./manual_monitoring/brdiv_monitoring_data.json")
    print("  - Plot: ./manual_monitoring/brdiv_monitoring_plot.png")


# =============================================================================
# EXAMPLE 5: Python - Analyze Results
# =============================================================================

def example_analyze_results():
    """Analyze monitoring data programmatically."""
    from teammate_generation.brdiv_monitoring_analysis import MonitoringDataAnalyzer
    
    # Load data
    analyzer = MonitoringDataAnalyzer("./exp1_results/brdiv_monitoring_data.json")
    
    # Print summary
    analyzer.print_summary()
    
    # Get statistics
    stats = analyzer.get_statistics_summary()
    print(f"\nBest SP return: {stats['sp_return_best']:.6f}")
    print(f"Best XP return: {stats['xp_return_best']:.6f}")
    print(f"Total time: {stats['total_training_time']:.2f}s")
    
    # Create custom plot
    analyzer.plot_with_annotations("./my_detailed_plot.png")
    
    # Export summary
    analyzer.save_summary_json("./my_summary.json")


# =============================================================================
# EXAMPLE 6: Python - Compare Multiple Runs
# =============================================================================

def example_compare_runs():
    """Compare results from multiple BRDiv runs."""
    from teammate_generation.brdiv_monitoring_analysis import compare_runs
    
    # Compare three runs
    compare_runs(
        data_files=[
            "./run1/brdiv_monitoring_data.json",
            "./run2/brdiv_monitoring_data.json",
            "./run3/brdiv_monitoring_data.json",
        ],
        labels=["Seed 1", "Seed 2", "Seed 3"]
    )


# =============================================================================
# EXAMPLE 7: Python - Custom Analysis
# =============================================================================

def example_custom_analysis():
    """Custom analysis of monitoring data."""
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Load raw data
    with open("./exp1_results/brdiv_monitoring_data.json") as f:
        data = json.load(f)
    
    times = np.array(data["wall_clock_times"])
    sp_returns = np.array(data["sp_returns"])
    xp_returns = np.array(data["xp_returns"])
    
    # Compute custom metrics
    print("=== Custom Analysis ===")
    print(f"Training duration: {times[-1]:.2f} seconds")
    print(f"Number of updates: {len(times)}")
    print(f"Avg time per update: {times[-1] / len(times):.4f}s")
    
    # Compute returns improvement
    sp_improvement = sp_returns[-1] - sp_returns[0]
    xp_improvement = xp_returns[-1] - xp_returns[0]
    print(f"SP return improvement: {sp_improvement:.6f}")
    print(f"XP return improvement: {xp_improvement:.6f}")
    
    # Compute when convergence happened (within 1% of final)
    sp_converged_at = None
    xp_converged_at = None
    sp_threshold = sp_returns[-1] * 0.99
    xp_threshold = xp_returns[-1] * 0.99
    
    for i, (sp, xp) in enumerate(zip(sp_returns, xp_returns)):
        if sp_converged_at is None and sp >= sp_threshold:
            sp_converged_at = (i, times[i])
        if xp_converged_at is None and xp >= xp_threshold:
            xp_converged_at = (i, times[i])
    
    if sp_converged_at:
        print(f"SP converged at update {sp_converged_at[0]} ({sp_converged_at[1]:.2f}s)")
    if xp_converged_at:
        print(f"XP converged at update {xp_converged_at[0]} ({xp_converged_at[1]:.2f}s)")
    
    # Create custom plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(times, sp_returns, 'o-', label="Self-play", linewidth=2)
    ax.plot(times, xp_returns, 's-', label="Cross-play", linewidth=2)
    ax.set_xlabel("Wall-clock Time (seconds)")
    ax.set_ylabel("Return")
    ax.set_title("BRDiv Training Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("./custom_plot.png", dpi=150)
    print("\nPlot saved to ./custom_plot.png")


# =============================================================================
# EXAMPLE 8: Full Workflow - Run and Analyze
# =============================================================================

def example_full_workflow():
    """Complete workflow: run BRDiv and analyze results."""
    import subprocess
    from pathlib import Path
    
    # Step 1: Run BRDiv with monitoring
    print("Step 1: Running BRDiv with monitoring...")
    result = subprocess.run([
        "python", "teammate_generation/run.py",
        "algorithm=brdiv/lbf",
        "task=lbf",
        "label=full_workflow_example",
        "enable_brdiv_monitoring=true",
        "brdiv_monitoring_dir=./workflow_results",
        "run_heldout_eval=false",
        "train_ego=false",
    ])
    
    if result.returncode != 0:
        print("BRDiv run failed!")
        return
    
    # Step 2: Analyze results
    print("\nStep 2: Analyzing results...")
    result = subprocess.run([
        "python", "teammate_generation/brdiv_monitoring_analysis.py",
        "./workflow_results/brdiv_monitoring_data.json",
        "--detailed"
    ])
    
    # Step 3: List outputs
    print("\nStep 3: Outputs created:")
    output_dir = Path("./workflow_results")
    for file in sorted(output_dir.glob("brdiv_monitoring_*")):
        print(f"  - {file}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("BRDiv Monitoring Examples")
        print()
        print("Usage: python examples.py <example_number>")
        print()
        print("Examples:")
        print("  1 - Basic monitoring via Python")
        print("  2 - Manual monitoring")
        print("  3 - Analyze results")
        print("  4 - Compare multiple runs")
        print("  5 - Custom analysis")
        print("  6 - Full workflow")
        print()
        print("Or run directly in Python:")
        print("  from examples import example_basic_monitoring")
        print("  example_basic_monitoring()")
        sys.exit(1)
    
    example_num = sys.argv[1]
    
    if example_num == "1":
        print("Note: Run this via Python import, not command line")
        print("See example_basic_monitoring()")
    elif example_num == "2":
        example_manual_monitoring()
    elif example_num == "3":
        example_analyze_results()
    elif example_num == "4":
        example_compare_runs()
    elif example_num == "5":
        example_custom_analysis()
    elif example_num == "6":
        example_full_workflow()
    else:
        print(f"Unknown example: {example_num}")
