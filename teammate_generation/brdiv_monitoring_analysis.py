"""
Post-processing and analysis utilities for BRDiv monitoring data.

This module provides utilities to load, analyze, and visualize BRDiv monitoring data.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt


class MonitoringDataAnalyzer:
    """Analyzer for BRDiv monitoring data."""
    
    def __init__(self, data_file: str):
        """Initialize analyzer with monitoring data file.
        
        Args:
            data_file: Path to brdiv_monitoring_data.json
        """
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        
        self.times = np.array(self.data["wall_clock_times"])
        self.steps = np.array(self.data["update_steps"])
        self.sp_returns = np.array(self.data["sp_returns"])
        self.xp_returns = np.array(self.data["xp_returns"])
    
    def get_training_duration(self) -> float:
        """Get total training time in seconds."""
        return self.times[-1] if len(self.times) > 0 else 0.0
    
    def get_num_updates(self) -> int:
        """Get number of training updates."""
        return len(self.steps)
    
    def get_convergence_rate(self) -> float:
        """Get updates per second during training."""
        duration = self.get_training_duration()
        return len(self.steps) / duration if duration > 0 else 0.0
    
    def get_return_improvement(self) -> Tuple[float, float]:
        """Get improvement in returns from start to end.
        
        Returns:
            Tuple of (sp_improvement, xp_improvement)
        """
        sp_improvement = self.sp_returns[-1] - self.sp_returns[0] if len(self.sp_returns) > 1 else 0.0
        xp_improvement = self.xp_returns[-1] - self.xp_returns[0] if len(self.xp_returns) > 1 else 0.0
        return sp_improvement, xp_improvement
    
    def get_best_returns(self) -> Tuple[Tuple[int, float, float], Tuple[int, float, float]]:
        """Get best returns and when they occurred.
        
        Returns:
            Tuple of ((sp_step, sp_time, sp_value), (xp_step, xp_time, xp_value))
        """
        best_sp_idx = int(np.argmax(self.sp_returns))
        best_xp_idx = int(np.argmax(self.xp_returns))
        
        sp_best = (self.steps[best_sp_idx], self.times[best_sp_idx], self.sp_returns[best_sp_idx])
        xp_best = (self.steps[best_xp_idx], self.times[best_xp_idx], self.xp_returns[best_xp_idx])
        
        return sp_best, xp_best
    
    def get_statistics_summary(self) -> Dict:
        """Get comprehensive statistics summary.
        
        Returns:
            Dictionary with various statistics
        """
        sp_imp, xp_imp = self.get_return_improvement()
        sp_best, xp_best = self.get_best_returns()
        
        return {
            "total_training_time": self.get_training_duration(),
            "num_updates": self.get_num_updates(),
            "convergence_rate": self.get_convergence_rate(),
            "sp_return_start": float(self.sp_returns[0]),
            "sp_return_end": float(self.sp_returns[-1]),
            "sp_return_improvement": sp_imp,
            "sp_return_best": float(sp_best[2]),
            "sp_return_best_at_step": int(sp_best[0]),
            "sp_return_best_at_time": float(sp_best[1]),
            "xp_return_start": float(self.xp_returns[0]),
            "xp_return_end": float(self.xp_returns[-1]),
            "xp_return_improvement": xp_imp,
            "xp_return_best": float(xp_best[2]),
            "xp_return_best_at_step": int(xp_best[0]),
            "xp_return_best_at_time": float(xp_best[1]),
        }
    
    def print_summary(self):
        """Print summary statistics to console."""
        stats = self.get_statistics_summary()
        
        print("\n" + "="*60)
        print("BRDiv Training Summary")
        print("="*60)
        print(f"Total training time: {stats['total_training_time']:.2f} seconds")
        print(f"Number of updates: {stats['num_updates']}")
        print(f"Convergence rate: {stats['convergence_rate']:.4f} updates/sec")
        print()
        print("Self-Play Returns:")
        print(f"  Start: {stats['sp_return_start']:.6f}")
        print(f"  End: {stats['sp_return_end']:.6f}")
        print(f"  Improvement: {stats['sp_return_improvement']:.6f}")
        print(f"  Best: {stats['sp_return_best']:.6f} (at step {stats['sp_return_best_at_step']}, {stats['sp_return_best_at_time']:.2f}s)")
        print()
        print("Cross-Play Returns:")
        print(f"  Start: {stats['xp_return_start']:.6f}")
        print(f"  End: {stats['xp_return_end']:.6f}")
        print(f"  Improvement: {stats['xp_return_improvement']:.6f}")
        print(f"  Best: {stats['xp_return_best']:.6f} (at step {stats['xp_return_best_at_step']}, {stats['xp_return_best_at_time']:.2f}s)")
        print("="*60 + "\n")
    
    def plot_with_annotations(self, output_file: Optional[str] = None):
        """Create detailed annotated plots.
        
        Args:
            output_file: Path to save plot (optional)
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Time vs Self-play returns
        axes[0, 0].plot(self.times, self.sp_returns, 'o-', linewidth=2, markersize=5, color='steelblue')
        best_sp_idx = int(np.argmax(self.sp_returns))
        axes[0, 0].plot(self.times[best_sp_idx], self.sp_returns[best_sp_idx], 'r*', markersize=15, label='Best')
        axes[0, 0].set_xlabel("Wall-clock Time (seconds)", fontsize=11)
        axes[0, 0].set_ylabel("Self-play Return", fontsize=11)
        axes[0, 0].set_title("Self-play Returns vs Time", fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Plot 2: Time vs Cross-play returns
        axes[0, 1].plot(self.times, self.xp_returns, 's-', linewidth=2, markersize=5, color='darkorange')
        best_xp_idx = int(np.argmax(self.xp_returns))
        axes[0, 1].plot(self.times[best_xp_idx], self.xp_returns[best_xp_idx], 'r*', markersize=15, label='Best')
        axes[0, 1].set_xlabel("Wall-clock Time (seconds)", fontsize=11)
        axes[0, 1].set_ylabel("Cross-play Return", fontsize=11)
        axes[0, 1].set_title("Cross-play Returns vs Time", fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # Plot 3: Update step vs Self-play returns
        axes[1, 0].plot(self.steps, self.sp_returns, 'o-', linewidth=2, markersize=5, color='steelblue')
        axes[1, 0].plot(self.steps[best_sp_idx], self.sp_returns[best_sp_idx], 'r*', markersize=15, label='Best')
        axes[1, 0].set_xlabel("Update Step", fontsize=11)
        axes[1, 0].set_ylabel("Self-play Return", fontsize=11)
        axes[1, 0].set_title("Self-play Returns vs Update Step", fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # Plot 4: Update step vs Cross-play returns
        axes[1, 1].plot(self.steps, self.xp_returns, 's-', linewidth=2, markersize=5, color='darkorange')
        axes[1, 1].plot(self.steps[best_xp_idx], self.xp_returns[best_xp_idx], 'r*', markersize=15, label='Best')
        axes[1, 1].set_xlabel("Update Step", fontsize=11)
        axes[1, 1].set_ylabel("Cross-play Return", fontsize=11)
        axes[1, 1].set_title("Cross-play Returns vs Update Step", fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved detailed plot to {output_file}")
        else:
            plt.show()
        
        plt.close()
    
    def save_summary_json(self, output_file: str):
        """Save statistics summary to JSON file.
        
        Args:
            output_file: Path to save JSON file
        """
        stats = self.get_statistics_summary()
        with open(output_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Saved summary to {output_file}")


def compare_runs(data_files: List[str], labels: Optional[List[str]] = None):
    """Compare multiple BRDiv runs.
    
    Args:
        data_files: List of paths to monitoring data files
        labels: Optional labels for each run
    """
    if labels is None:
        labels = [f"Run {i+1}" for i in range(len(data_files))]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for data_file, label in zip(data_files, labels):
        analyzer = MonitoringDataAnalyzer(data_file)
        
        axes[0].plot(analyzer.times, analyzer.sp_returns, 'o-', label=label, linewidth=2, markersize=4)
        axes[1].plot(analyzer.times, analyzer.xp_returns, 's-', label=label, linewidth=2, markersize=4)
    
    axes[0].set_xlabel("Wall-clock Time (seconds)", fontsize=11)
    axes[0].set_ylabel("Self-play Return", fontsize=11)
    axes[0].set_title("Self-play Returns Comparison", fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    axes[1].set_xlabel("Wall-clock Time (seconds)", fontsize=11)
    axes[1].set_ylabel("Cross-play Return", fontsize=11)
    axes[1].set_title("Cross-play Returns Comparison", fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python brdiv_monitoring_analysis.py <data_file.json> [--detailed] [--summary]")
        print()
        print("Examples:")
        print("  # Print summary")
        print("  python brdiv_monitoring_analysis.py brdiv_monitoring_data.json")
        print()
        print("  # Print detailed summary and create plots")
        print("  python brdiv_monitoring_analysis.py brdiv_monitoring_data.json --detailed")
        sys.exit(1)
    
    data_file = sys.argv[1]
    analyzer = MonitoringDataAnalyzer(data_file)
    
    # Print summary
    analyzer.print_summary()
    
    # Create detailed plots if requested
    if "--detailed" in sys.argv:
        output_dir = Path(data_file).parent
        output_file = output_dir / "brdiv_monitoring_detailed_plot.png"
        analyzer.plot_with_annotations(str(output_file))
    
    # Save summary if requested
    if "--summary" in sys.argv:
        output_dir = Path(data_file).parent
        summary_file = output_dir / "brdiv_monitoring_summary.json"
        analyzer.save_summary_json(str(summary_file))
