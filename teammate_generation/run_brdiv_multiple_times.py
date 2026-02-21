#!/usr/bin/env python
"""
Script to run BRDiv algorithm multiple times and aggregate the collected data.

This script enables running BRDiv with monitoring multiple times and automatically
aggregates the timing and returns data into a single dataset for analysis.

Features:
- Runs BRDiv multiple times with different seeds
- Aggregates wall-clock times and returns from all runs
- Generates combined plots showing all data points
- Saves aggregated data to a single JSON file
- Supports both CLI and programmatic usage

Usage from command line:
    python teammate_generation/run_brdiv_multiple_times.py \
        algorithm=brdiv/lbf \
        task=lbf \
        label=test_brdiv_aggregated \
        enable_brdiv_monitoring=true \
        brdiv_monitoring_dir=./brdiv_results \
        num_runs=5 \
        run_heldout_eval=false \
        train_ego=false

Usage from Python:
    from teammate_generation.run_brdiv_multiple_times import run_brdiv_multiple_times_aggregated
    
    aggregated_data = run_brdiv_multiple_times_aggregated(
        cfg_dict,
        wandb_logger,
        num_runs=5,
        aggregation_output_dir="./aggregated_results"
    )
"""

import hydra
import json
import logging
import shutil
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from omegaconf import OmegaConf
from typing import Dict, List, Tuple, Any
import time

from evaluation.heldout_eval import run_heldout_evaluation, log_heldout_metrics
from common.plot_utils import get_metric_names
from common.wandb_visualizations import Logger
from teammate_generation.BRDiv import run_brdiv
from teammate_generation.LBRDiv import run_lbrdiv
from teammate_generation.CoMeDi import run_comedi
from teammate_generation.fcp import run_fcp
from teammate_generation.train_ego import train_ego_agent

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class BRDivAggregator:
    """Aggregates monitoring data from multiple BRDiv runs."""
    
    def __init__(self, output_dir: str = "./brdiv_aggregated"):
        """Initialize the aggregator.
        
        Args:
            output_dir: Directory where to save aggregated data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.aggregated_data = {
            "wall_clock_times": [],
            "update_steps": [],
            "sp_returns": [],
            "xp_returns": [],
            "run_ids": [],  # Track which run each data point came from
        }
        self.individual_runs = []  # Store individual run data
    
    def add_run(self, run_data: Dict[str, List[float]], run_id: int):
        """Add monitoring data from a single run.
        
        Args:
            run_data: Dictionary with keys 'wall_clock_times', 'update_steps', 'sp_returns', 'xp_returns'
            run_id: Identifier for this run
        """
        # Store individual run data
        self.individual_runs.append({
            "run_id": run_id,
            "data": run_data
        })
        
        # Add to aggregated data
        num_points = len(run_data["update_steps"])
        self.aggregated_data["wall_clock_times"].extend(run_data["wall_clock_times"])
        self.aggregated_data["update_steps"].extend(run_data["update_steps"])
        self.aggregated_data["sp_returns"].extend(run_data["sp_returns"])
        self.aggregated_data["xp_returns"].extend(run_data["xp_returns"])
        self.aggregated_data["run_ids"].extend([run_id] * num_points)
        
        log.info(f"Added run {run_id} with {num_points} data points")
    
    def load_run_from_file(self, json_file: str, run_id: int):
        """Load monitoring data from a JSON file and add it to aggregation.
        
        Args:
            json_file: Path to brdiv_monitoring_data.json
            run_id: Identifier for this run
        """
        with open(json_file, 'r') as f:
            run_data = json.load(f)
        self.add_run(run_data, run_id)
    
    def save_aggregated_data(self, filename: str = "brdiv_aggregated_data.json"):
        """Save the aggregated data to a JSON file.
        
        Args:
            filename: Name of the output file
        """
        filepath = self.output_dir / filename
        data_serializable = {
            k: [float(v) if isinstance(v, (np.floating, np.integer)) else v 
                 for v in vals]
            for k, vals in self.aggregated_data.items()
        }
        with open(filepath, "w") as f:
            json.dump(data_serializable, f, indent=2)
        log.info(f"Aggregated data saved to {filepath}")
        
        # Also save summary statistics
        summary = self._compute_summary_statistics()
        summary_file = self.output_dir / "aggregated_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        log.info(f"Summary statistics saved to {summary_file}")
        
        return filepath
    
    def _compute_summary_statistics(self) -> Dict[str, Any]:
        """Compute summary statistics from aggregated data.
        
        Returns:
            Dictionary with summary statistics
        """
        sp_returns = np.array(self.aggregated_data["sp_returns"])
        xp_returns = np.array(self.aggregated_data["xp_returns"])
        wall_clock_times = np.array(self.aggregated_data["wall_clock_times"])
        
        summary = {
            "num_runs": len(self.individual_runs),
            "num_total_data_points": len(sp_returns),
            "sp_returns": {
                "mean": float(sp_returns.mean()),
                "std": float(sp_returns.std()),
                "min": float(sp_returns.min()),
                "max": float(sp_returns.max()),
            },
            "xp_returns": {
                "mean": float(xp_returns.mean()),
                "std": float(xp_returns.std()),
                "min": float(xp_returns.min()),
                "max": float(xp_returns.max()),
            },
            "wall_clock_times": {
                "mean": float(wall_clock_times.mean()),
                "std": float(wall_clock_times.std()),
                "min": float(wall_clock_times.min()),
                "max": float(wall_clock_times.max()),
            }
        }
        return summary
    
    def plot_aggregated_results(self, filename: str = "brdiv_aggregated_plot.png"):
        """Plot aggregated wall-clock time vs returns with individual runs connected by lines.
        
        Args:
            filename: Name of the output plot file
        """
        if not self.aggregated_data["wall_clock_times"]:
            log.warning("No data to plot")
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Use different colors for each run
        run_ids = np.array(self.aggregated_data["run_ids"])
        unique_runs = np.unique(run_ids)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_runs)))
        
        times = np.array(self.aggregated_data["wall_clock_times"])
        sp_returns = np.array(self.aggregated_data["sp_returns"])
        xp_returns = np.array(self.aggregated_data["xp_returns"])
        
        # Plot 1: Wall-clock time vs Self-play returns (connected lines)
        for idx, run_id in enumerate(unique_runs):
            mask = run_ids == run_id
            times_run = times[mask]
            sp_ret_run = sp_returns[mask]
            # Sort by time to connect points properly
            sort_idx = np.argsort(times_run)
            axes[0].plot(
                times_run[sort_idx],
                sp_ret_run[sort_idx],
                linewidth=1.5,
                color=colors[idx],
                label=f'Run {run_id}',
                alpha=0.8
            )
        
        axes[0].set_xlabel("Wall-clock Time (seconds)", fontsize=12)
        axes[0].set_ylabel("Self-play Return", fontsize=12)
        axes[0].set_title("BRDiv: Self-play Returns vs Time", fontsize=13, fontweight='bold')
        axes[0].set_xscale('log')
        axes[0].set_xticks([10, 100, 1000])
        axes[0].grid(True, alpha=0.3, which='both')
        axes[0].legend(fontsize=9, loc='best')
        
        # Plot 2: Wall-clock time vs Cross-play returns (connected lines)
        for idx, run_id in enumerate(unique_runs):
            mask = run_ids == run_id
            times_run = times[mask]
            xp_ret_run = xp_returns[mask]
            # Sort by time to connect points properly
            sort_idx = np.argsort(times_run)
            axes[1].plot(
                times_run[sort_idx],
                xp_ret_run[sort_idx],
                linewidth=1.5,
                color=colors[idx],
                label=f'Run {run_id}',
                alpha=0.8
            )
        
        axes[1].set_xlabel("Wall-clock Time (seconds)", fontsize=12)
        axes[1].set_ylabel("Cross-play Return", fontsize=12)
        axes[1].set_title("BRDiv: Cross-play Returns vs Time", fontsize=13, fontweight='bold')
        axes[1].set_xscale('log')
        axes[1].set_xticks([10, 100, 1000])
        axes[1].grid(True, alpha=0.3, which='both')
        axes[1].legend(fontsize=9, loc='best')
        
        plt.tight_layout()
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        log.info(f"Aggregated plot saved to {filepath}")
        plt.close()
        return filepath
    
    def plot_aggregated_results_combined(self, filename: str = "brdiv_aggregated_combined_plot.png"):
        """Plot aggregated results with individual runs connected by lines.
        
        Args:
            filename: Name of the output plot file
        """
        if not self.aggregated_data["wall_clock_times"]:
            log.warning("No data to plot")
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot individual runs as connected lines
        run_ids = np.array(self.aggregated_data["run_ids"])
        unique_runs = np.unique(run_ids)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_runs)))
        
        times = np.array(self.aggregated_data["wall_clock_times"])
        sp_returns = np.array(self.aggregated_data["sp_returns"])
        xp_returns = np.array(self.aggregated_data["xp_returns"])
        
        # Plot 1: Self-play returns with connected lines
        for idx, run_id in enumerate(unique_runs):
            mask = run_ids == run_id
            times_run = times[mask]
            sp_ret_run = sp_returns[mask]
            sort_idx = np.argsort(times_run)
            axes[0].plot(times_run[sort_idx], sp_ret_run[sort_idx], linewidth=1.5, color=colors[idx], label=f'Run {int(run_id)}', alpha=0.8)
        
        axes[0].set_xlabel("Wall-clock Time (seconds)", fontsize=12)
        axes[0].set_ylabel("Self-play Return", fontsize=12)
        axes[0].set_title("BRDiv: Self-play Returns vs Time", fontsize=13, fontweight='bold')
        axes[0].set_xscale('log')
        axes[0].set_xticks([10, 100, 1000])
        axes[0].grid(True, alpha=0.3, which='both')
        axes[0].legend(fontsize=9)
        
        # Plot 2: Cross-play returns with connected lines
        for idx, run_id in enumerate(unique_runs):
            mask = run_ids == run_id
            times_run = times[mask]
            xp_ret_run = xp_returns[mask]
            sort_idx = np.argsort(times_run)
            axes[1].plot(times_run[sort_idx], xp_ret_run[sort_idx], linewidth=1.5, color=colors[idx], label=f'Run {int(run_id)}', alpha=0.8)
        
        axes[1].set_xlabel("Wall-clock Time (seconds)", fontsize=12)
        axes[1].set_ylabel("Cross-play Return", fontsize=12)
        axes[1].set_title("BRDiv: Cross-play Returns vs Time", fontsize=13, fontweight='bold')
        axes[1].set_xscale('log')
        axes[1].set_xticks([10, 100, 1000])
        axes[1].grid(True, alpha=0.3, which='both')
        axes[1].legend(fontsize=9)
        
        plt.tight_layout()
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        log.info(f"Combined plot saved to {filepath}")
        plt.close()
        return filepath


def run_brdiv_multiple_times_aggregated(
    cfg_dict: Dict[str, Any],
    wandb_logger,
    num_runs: int = 3,
    aggregation_output_dir: str = "./brdiv_aggregated",
    individual_dirs_base: str = "./brdiv_runs",
) -> Tuple[Dict[str, Any], BRDivAggregator]:
    """Run BRDiv multiple times and aggregate the results.
    
    Args:
        cfg_dict: Configuration dictionary
        wandb_logger: Weights & Biases logger
        num_runs: Number of times to run BRDiv
        aggregation_output_dir: Directory to save aggregated results
        individual_dirs_base: Base directory for individual run outputs
    
    Returns:
        Tuple of (aggregated_data, aggregator)
    """
    aggregator = BRDivAggregator(output_dir=aggregation_output_dir)
    algorithm = cfg_dict["algorithm"]["ALG"]
    
    # Track individual run output directories
    run_output_dirs = []
    
    # Get base seed for generating different seeds for each run
    base_seed = cfg_dict.get("algorithm", {}).get("TRAIN_SEED", 0)
    
    log.info(f"Starting {num_runs} runs of {algorithm}")
    
    for run_idx in range(num_runs):
        log.info(f"\n{'='*60}")
        log.info(f"Run {run_idx + 1}/{num_runs}")
        log.info(f"{'='*60}\n")
        
        # Create a unique directory for each run
        run_output_dir = Path(individual_dirs_base) / f"run_{run_idx}"
        run_output_dir.mkdir(parents=True, exist_ok=True)
        run_output_dirs.append(run_output_dir)
        
        # Update config to save results in this run's directory and use different seed
        cfg_dict_run = cfg_dict.copy()
        cfg_dict_run["brdiv_monitoring_dir"] = str(run_output_dir / "monitoring")
        
        # Generate a unique seed for this run by adding the run index to the base seed
        unique_seed = base_seed + run_idx
        cfg_dict_run["algorithm"] = cfg_dict_run.get("algorithm", {}).copy()
        cfg_dict_run["algorithm"]["TRAIN_SEED"] = unique_seed
        
        # Increase NUM_CHECKPOINTS to match NUM_UPDATES so we get an evaluation at every step
        # This gives us more data points for monitoring instead of just checkpoint intervals
        num_timesteps = cfg_dict_run["algorithm"].get("TOTAL_TIMESTEPS", 4.5e7)
        num_envs = cfg_dict_run["algorithm"].get("NUM_ENVS", 64)
        rollout_length = cfg_dict_run["algorithm"].get("ROLLOUT_LENGTH", 128)
        num_agents = 2  # BRDiv always has 2 agents
        num_updates = int(num_timesteps // (num_agents * rollout_length * num_envs))
        cfg_dict_run["algorithm"]["NUM_CHECKPOINTS"] = max(num_updates, 1)  # Evaluate at every update
        
        log.info(f"Using TRAIN_SEED={unique_seed} for run {run_idx}")
        log.info(f"Set NUM_CHECKPOINTS={cfg_dict_run['algorithm']['NUM_CHECKPOINTS']} to match NUM_UPDATES for full monitoring coverage")
        
        # For runs after the first, remove existing saved_train_run to avoid Hydra conflicts
        # since multiple runs use the same Hydra output directory
        if run_idx > 0:
            import hydra
            try:
                hydra_output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
                saved_train_run_dir = hydra_output_dir / "saved_train_run"
                if saved_train_run_dir.exists():
                    log.info(f"Removing existing checkpoint directory for intermediate run: {saved_train_run_dir}")
                    shutil.rmtree(saved_train_run_dir)
            except Exception as e:
                log.warning(f"Could not remove saved_train_run directory: {e}")
        
        # Run the algorithm
        if algorithm == "brdiv":
            partner_params, partner_population = run_brdiv(cfg_dict_run, wandb_logger)
        elif algorithm == "fcp":
            partner_params, partner_population = run_fcp(cfg_dict_run, wandb_logger)
        elif algorithm == "lbrdiv":
            partner_params, partner_population = run_lbrdiv(cfg_dict_run, wandb_logger)
        elif algorithm == "comedi":
            partner_params, partner_population = run_comedi(cfg_dict_run, wandb_logger)
        else:
            raise NotImplementedError(f"Algorithm {algorithm} not implemented.")
        
        # Try to load monitoring data from this run
        monitoring_file = run_output_dir / "monitoring" / "brdiv_monitoring_data.json"
        if monitoring_file.exists():
            log.info(f"Loading monitoring data from {monitoring_file}")
            aggregator.load_run_from_file(str(monitoring_file), run_id=run_idx)
        else:
            log.warning(f"Monitoring data not found at {monitoring_file}")
    
    # Save and plot aggregated results
    log.info(f"\n{'='*60}")
    log.info(f"Aggregating results from {num_runs} runs")
    log.info(f"{'='*60}\n")
    
    aggregator.save_aggregated_data()
    aggregator.plot_aggregated_results()
    aggregator.plot_aggregated_results_combined()
    
    log.info(f"\nAggregated results saved to: {aggregator.output_dir}")
    
    return aggregator.aggregated_data, aggregator


@hydra.main(version_base=None, config_path="configs", config_name="base_config_teammate")
def run_training_multiple(cfg):
    """Run BRDiv multiple times with monitoring enabled."""
    print(OmegaConf.to_yaml(cfg, resolve=True))
    
    # Convert to container to enable monitoring parameters
    cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    
    # Enable monitoring for BRDiv
    cfg_dict["enable_brdiv_monitoring"] = cfg_dict.get("enable_brdiv_monitoring", True)
    
    # Get number of runs from config (default: 3)
    num_runs = cfg_dict.get("num_runs", 3)
    aggregation_output_dir = cfg_dict.get("aggregation_output_dir", "./brdiv_aggregated_results")
    individual_dirs_base = cfg_dict.get("individual_dirs_base", "./brdiv_individual_runs")
    
    wandb_logger = Logger(cfg)
    
    # Run BRDiv multiple times and aggregate
    aggregated_data, aggregator = run_brdiv_multiple_times_aggregated(
        cfg_dict,
        wandb_logger,
        num_runs=num_runs,
        aggregation_output_dir=aggregation_output_dir,
        individual_dirs_base=individual_dirs_base,
    )
    
    wandb_logger.close()
    
    print(f"\n{'='*60}")
    print(f"AGGREGATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total data points collected: {len(aggregated_data['wall_clock_times'])}")
    print(f"Output directory: {aggregation_output_dir}")
    print(f"Individual runs: {individual_dirs_base}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    run_training_multiple()
