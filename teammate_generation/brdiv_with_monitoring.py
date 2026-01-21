"""
Wrapper for BRDiv that records timing and returns data during training.
This module wraps the BRDiv algorithm to track:
- Wall-clock time since algorithm start
- Returns at each evaluation
And stores this data to a file and plots it after training completes.
"""

import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
from functools import wraps

log = logging.getLogger(__name__)


class BRDivMonitor:
    """Monitors BRDiv training and records timing and returns data."""
    
    def __init__(self, output_dir: str = "./brdiv_monitoring"):
        """Initialize the monitor.
        
        Args:
            output_dir: Directory where to save monitoring data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.start_time = None
        self.data = {
            "wall_clock_times": [],
            "update_steps": [],
            "sp_returns": [],  # Self-play returns
            "xp_returns": [],  # Cross-play returns
        }
    
    def start(self):
        """Start the timer."""
        self.start_time = time.time()
        log.info(f"BRDiv monitoring started at {self.start_time}")
    
    def record_update(self, update_step: int, sp_return: float, xp_return: float):
        """Record metrics at a specific update step.
        
        Args:
            update_step: The update step number
            sp_return: Self-play return (confederate vs confederate)
            xp_return: Cross-play return (confederate vs best response)
        """
        elapsed_time = time.time() - self.start_time
        self.data["wall_clock_times"].append(elapsed_time)
        self.data["update_steps"].append(update_step)
        self.data["sp_returns"].append(float(sp_return))
        self.data["xp_returns"].append(float(xp_return))
        
        log.info(
            f"Update {update_step}: "
            f"Elapsed time: {elapsed_time:.2f}s, "
            f"SP return: {sp_return:.4f}, "
            f"XP return: {xp_return:.4f}"
        )
    
    def record_update_with_time(self, update_step: int, sp_return: float, xp_return: float, elapsed_time: float):
        """Record metrics at a specific update step with explicit elapsed time.
        
        This is useful when the actual wall-clock time is known (e.g., reconstructed from training time).
        
        Args:
            update_step: The update step number
            sp_return: Self-play return (confederate vs confederate)
            xp_return: Cross-play return (confederate vs best response)
            elapsed_time: Elapsed time in seconds since algorithm start
        """
        self.data["wall_clock_times"].append(elapsed_time)
        self.data["update_steps"].append(update_step)
        self.data["sp_returns"].append(float(sp_return))
        self.data["xp_returns"].append(float(xp_return))
        
        log.info(
            f"Update {update_step}: "
            f"Elapsed time: {elapsed_time:.2f}s, "
            f"SP return: {sp_return:.4f}, "
            f"XP return: {xp_return:.4f}"
        )
    
    def save_data(self, filename: str = "brdiv_monitoring_data.json"):
        """Save the collected data to a JSON file.
        
        Args:
            filename: Name of the output file
        """
        filepath = self.output_dir / filename
        # Convert numpy types to native Python types for JSON serialization
        data_serializable = {
            k: [float(v) if isinstance(v, (np.floating, np.integer)) else v 
                 for v in vals]
            for k, vals in self.data.items()
        }
        with open(filepath, "w") as f:
            json.dump(data_serializable, f, indent=2)
        log.info(f"Monitoring data saved to {filepath}")
        return filepath
    
    def plot_results(self, filename: str = "brdiv_monitoring_plot.png"):
        """Plot wall-clock time vs returns.
        
        Args:
            filename: Name of the output plot file
        """
        if not self.data["wall_clock_times"]:
            log.warning("No data to plot")
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Wall-clock time vs Self-play returns
        axes[0].plot(
            self.data["wall_clock_times"],
            self.data["sp_returns"],
            marker='o',
            linewidth=2,
            markersize=6,
            label='Self-play returns'
        )
        axes[0].set_xlabel("Wall-clock Time (seconds)", fontsize=12)
        axes[0].set_ylabel("Self-play Return", fontsize=12)
        axes[0].set_title("BRDiv Training: Self-play Returns vs Time", fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Plot 2: Wall-clock time vs Cross-play returns
        axes[1].plot(
            self.data["wall_clock_times"],
            self.data["xp_returns"],
            marker='s',
            linewidth=2,
            markersize=6,
            color='orange',
            label='Cross-play returns'
        )
        axes[1].set_xlabel("Wall-clock Time (seconds)", fontsize=12)
        axes[1].set_ylabel("Cross-play Return", fontsize=12)
        axes[1].set_title("BRDiv Training: Cross-play Returns vs Time", fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        plt.tight_layout()
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        log.info(f"Plot saved to {filepath}")
        plt.close()
        return filepath


def wrap_run_brdiv_with_monitoring(run_brdiv_fn):
    """
    Decorator to wrap run_brdiv with monitoring capabilities.
    
    This extracts timing and returns data from the training output and
    records it using BRDivMonitor.
    
    Usage:
        from teammate_generation.brdiv_with_monitoring import wrap_run_brdiv_with_monitoring
        from teammate_generation.BRDiv import run_brdiv
        
        monitored_run_brdiv = wrap_run_brdiv_with_monitoring(run_brdiv)
        partner_params, partner_population = monitored_run_brdiv(config, wandb_logger)
    """
    @wraps(run_brdiv_fn)
    def wrapper(config, wandb_logger):
        monitor = BRDivMonitor(
            output_dir=config.get("monitoring_dir", "./brdiv_monitoring")
        )
        monitor.start()
        
        # Run the original BRDiv algorithm
        partner_params, partner_population = run_brdiv_fn(config, wandb_logger)
        
        return partner_params, partner_population, monitor
    
    return wrapper


def extract_and_record_metrics(
    monitor: BRDivMonitor,
    metrics: Dict[str, Any],
    pop_size: int
):
    """
    Extract metrics from BRDiv output and record them in the monitor.
    
    Args:
        monitor: BRDivMonitor instance
        metrics: Metrics dictionary from BRDiv training
        pop_size: Population size used in training
    """
    from teammate_generation.BRDiv import _get_all_ids
    
    # Extract shape information
    num_seeds, num_updates, _, _, _ = metrics["pg_loss_conf_agent"].shape
    
    # Get all returns - shape (num_seeds, num_updates, (pop_size)^2, num_eval_episodes, num_agents_per_game)
    all_returns = np.asarray(metrics["eval_ep_last_info"]["returned_episode_returns"])
    
    # Separate self-play and cross-play returns
    all_conf_ids, all_br_ids = _get_all_ids(pop_size)
    sp_mask = (all_conf_ids == all_br_ids)
    sp_returns = all_returns[:, :, sp_mask]
    xp_returns = all_returns[:, :, ~sp_mask]
    
    # Average over seeds, pairs, episodes, and agents
    sp_return_curve = sp_returns.mean(axis=(0, 2, 3, 4))
    xp_return_curve = xp_returns.mean(axis=(0, 2, 3, 4))
    
    # Record at each update step
    for update_step in range(num_updates):
        monitor.record_update(
            update_step=update_step,
            sp_return=sp_return_curve[update_step],
            xp_return=xp_return_curve[update_step]
        )
    
    return monitor


# Example usage function
def run_brdiv_with_monitoring(config, wandb_logger, monitoring_dir: str = "./brdiv_monitoring"):
    """
    Run BRDiv with integrated monitoring of timing and returns.
    
    Args:
        config: Configuration dictionary
        wandb_logger: Weights & Biases logger
        monitoring_dir: Directory to save monitoring data
    
    Returns:
        Tuple of (partner_params, partner_population, monitor)
    """
    from teammate_generation.BRDiv import run_brdiv
    
    config["monitoring_dir"] = monitoring_dir
    
    monitored_fn = wrap_run_brdiv_with_monitoring(run_brdiv)
    partner_params, partner_population, monitor = monitored_fn(config, wandb_logger)
    
    # Note: To use this, you would need to capture metrics from the training run
    # and call extract_and_record_metrics(monitor, metrics, pop_size)
    # This requires modifying log_metrics() in BRDiv.py to pass metrics to monitoring
    
    # Save and plot
    monitor.save_data()
    monitor.plot_results()
    
    return partner_params, partner_population, monitor
