"""
Monitoring system for PPO ego agent training.
Records wall-clock time and returns during training and saves to JSON with plots.
"""

import time
import json
import logging
from pathlib import Path
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)


class PPOMonitor:
    """Monitors PPO ego agent training and records timing and returns data."""
    
    def __init__(self, output_dir: str = "./ppo_monitoring"):
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
            "ego_returns": [],
            "ego_value_loss": [],
            "ego_actor_loss": [],
            "ego_entropy_loss": [],
        }
    
    def start(self):
        """Start the timer."""
        self.start_time = time.time()
        log.info(f"PPO monitoring started at {self.start_time}")
    
    def record_update_with_time(self, update_step: int, ego_return: float, 
                                 value_loss: float = None, actor_loss: float = None,
                                 entropy_loss: float = None, elapsed_time: float = None):
        """Record metrics at a specific update step with explicit elapsed time.
        
        Args:
            update_step: The update step number
            ego_return: Ego agent return
            value_loss: Value loss (optional)
            actor_loss: Actor loss (optional)
            entropy_loss: Entropy loss (optional)
            elapsed_time: Elapsed time (if None, uses real wall-clock time)
        """
        if elapsed_time is None:
            elapsed_time = time.time() - self.start_time if self.start_time else 0
        
        self.data["wall_clock_times"].append(elapsed_time)
        self.data["update_steps"].append(update_step)
        self.data["ego_returns"].append(float(ego_return))
        self.data["ego_value_loss"].append(float(value_loss) if value_loss is not None else None)
        self.data["ego_actor_loss"].append(float(actor_loss) if actor_loss is not None else None)
        self.data["ego_entropy_loss"].append(float(entropy_loss) if entropy_loss is not None else None)
        
        log.info(
            f"Update {update_step}: "
            f"Elapsed time: {elapsed_time:.2f}s, "
            f"Ego return: {ego_return:.4f}"
        )
    
    def save_data(self, filename: str = "ppo_monitoring_data.json"):
        """Save the collected data to a JSON file.
        
        Args:
            filename: Name of the output file
        """
        filepath = self.output_dir / filename
        # Convert numpy types to native Python types for JSON serialization
        data_serializable = {
            k: [float(v) if isinstance(v, (np.floating, np.integer)) and v is not None else v
                 for v in vals]
            for k, vals in self.data.items()
        }
        with open(filepath, "w") as f:
            json.dump(data_serializable, f, indent=2)
        log.info(f"PPO monitoring data saved to {filepath}")
        return filepath
    
    def plot_results(self, filename: str = "ppo_monitoring_plot.png"):
        """Plot wall-clock time vs returns and losses.
        
        Args:
            filename: Name of the output plot file
        """
        if not self.data["wall_clock_times"]:
            log.warning("No data to plot")
            return None
        
        times = np.array(self.data["wall_clock_times"])
        returns = np.array(self.data["ego_returns"])
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Wall-clock time vs Ego returns
        axes[0, 0].plot(
            times, returns,
            marker='o',
            linewidth=2,
            markersize=5,
            color='steelblue',
            label='Ego return'
        )
        axes[0, 0].set_xlabel("Wall-clock Time (seconds)", fontsize=11)
        axes[0, 0].set_ylabel("Ego Return", fontsize=11)
        axes[0, 0].set_title("PPO Training: Ego Returns vs Time", fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Plot 2: Update step vs Ego returns
        steps = np.array(self.data["update_steps"])
        axes[0, 1].plot(
            steps, returns,
            marker='o',
            linewidth=2,
            markersize=5,
            color='steelblue',
            label='Ego return'
        )
        axes[0, 1].set_xlabel("Update Step", fontsize=11)
        axes[0, 1].set_ylabel("Ego Return", fontsize=11)
        axes[0, 1].set_title("PPO Training: Ego Returns vs Update Step", fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # Plot 3: Losses over time
        value_losses = np.array([x for x in self.data["ego_value_loss"] if x is not None])
        actor_losses = np.array([x for x in self.data["ego_actor_loss"] if x is not None])
        entropy_losses = np.array([x for x in self.data["ego_entropy_loss"] if x is not None])
        
        if len(value_losses) > 0:
            axes[1, 0].plot(times[:len(value_losses)], value_losses, marker='o', label='Value Loss', linewidth=2, markersize=4)
        if len(actor_losses) > 0:
            axes[1, 0].plot(times[:len(actor_losses)], actor_losses, marker='s', label='Actor Loss', linewidth=2, markersize=4)
        if len(entropy_losses) > 0:
            axes[1, 0].plot(times[:len(entropy_losses)], entropy_losses, marker='^', label='Entropy Loss', linewidth=2, markersize=4)
        
        axes[1, 0].set_xlabel("Wall-clock Time (seconds)", fontsize=11)
        axes[1, 0].set_ylabel("Loss", fontsize=11)
        axes[1, 0].set_title("PPO Training: Losses vs Time", fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # Plot 4: Return improvement over time
        if len(returns) > 1:
            return_improvement = returns - returns[0]
            axes[1, 1].plot(
                times, return_improvement,
                marker='o',
                linewidth=2,
                markersize=5,
                color='darkgreen',
                label='Return improvement'
            )
            axes[1, 1].set_xlabel("Wall-clock Time (seconds)", fontsize=11)
            axes[1, 1].set_ylabel("Return Improvement", fontsize=11)
            axes[1, 1].set_title("PPO Training: Return Improvement vs Time", fontsize=12, fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].legend()
        
        plt.tight_layout()
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        log.info(f"Plot saved to {filepath}")
        plt.close()
        return filepath
