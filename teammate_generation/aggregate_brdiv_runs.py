#!/usr/bin/env python
"""
Utility script to aggregate BRDiv monitoring data from multiple existing runs.

This script is useful when you already have multiple BRDiv run directories and want to:
1. Aggregate their monitoring data into a single file
2. Generate combined plots showing all data points
3. Compute summary statistics across runs

Usage:
    # Aggregate runs from a directory structure
    python teammate_generation/aggregate_brdiv_runs.py \
        --input-dir ./brdiv_runs \
        --output-dir ./brdiv_aggregated \
        --pattern "run_*"
    
    # Or specify individual run directories
    python teammate_generation/aggregate_brdiv_runs.py \
        --run-dirs ./brdiv_runs/run_0 ./brdiv_runs/run_1 ./brdiv_runs/run_2 \
        --output-dir ./brdiv_aggregated

    # With glob pattern
    python teammate_generation/aggregate_brdiv_runs.py \
        --input-dir ./results \
        --output-dir ./aggregated \
        --pattern "**/monitoring/brdiv_monitoring_data.json"

Features:
- Automatically finds brdiv_monitoring_data.json files in directory structures
- Supports glob patterns for flexible file discovery
- Preserves individual run information in the aggregated data
- Generates comparison plots with different colors per run
- Computes and saves summary statistics
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


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
        
        log.info(f"✓ Added run {run_id} with {num_points} data points")
    
    def load_run_from_file(self, json_file: str, run_id: int):
        """Load monitoring data from a JSON file and add it to aggregation.
        
        Args:
            json_file: Path to brdiv_monitoring_data.json
            run_id: Identifier for this run
        """
        json_path = Path(json_file)
        if not json_path.exists():
            log.warning(f"✗ File not found: {json_file}")
            return False
        
        try:
            with open(json_file, 'r') as f:
                run_data = json.load(f)
            self.add_run(run_data, run_id)
            return True
        except Exception as e:
            log.error(f"✗ Failed to load {json_file}: {e}")
            return False
    
    def find_monitoring_files(self, search_dir: str, pattern: str = "**/monitoring/brdiv_monitoring_data.json") -> List[Path]:
        """Find all monitoring data files in a directory using glob pattern.
        
        Args:
            search_dir: Directory to search in
            pattern: Glob pattern to match files
        
        Returns:
            List of Path objects matching the pattern
        """
        search_path = Path(search_dir)
        matching_files = list(search_path.glob(pattern))
        log.info(f"Found {len(matching_files)} monitoring files matching pattern '{pattern}'")
        return sorted(matching_files)
    
    def load_runs_from_directory(self, input_dir: str, pattern: str = "**/monitoring/brdiv_monitoring_data.json"):
        """Load all monitoring data files from a directory.
        
        Args:
            input_dir: Directory to search for monitoring files
            pattern: Glob pattern for finding files
        """
        files = self.find_monitoring_files(input_dir, pattern)
        
        for run_id, file_path in enumerate(files):
            self.load_run_from_file(str(file_path), run_id)
    
    def load_runs_from_dirs(self, run_dirs: List[str]):
        """Load monitoring data from specific run directories.
        
        Args:
            run_dirs: List of directories containing monitoring data
        """
        for run_id, run_dir in enumerate(run_dirs):
            monitoring_file = Path(run_dir) / "monitoring" / "brdiv_monitoring_data.json"
            if not monitoring_file.exists():
                # Try alternative location
                monitoring_file = Path(run_dir) / "brdiv_monitoring_data.json"
            
            if monitoring_file.exists():
                self.load_run_from_file(str(monitoring_file), run_id)
            else:
                log.warning(f"✗ No monitoring data found in {run_dir}")
    
    def save_aggregated_data(self, filename: str = "brdiv_aggregated_data.json"):
        """Save the aggregated data to a JSON file.
        
        Args:
            filename: Name of the output file
        """
        filepath = self.output_dir / filename
        data_serializable = {
            k: [float(v) if isinstance(v, (np.floating, np.integer)) else int(v) if isinstance(v, (np.integer, int)) else v
                 for v in vals]
            for k, vals in self.aggregated_data.items()
        }
        with open(filepath, "w") as f:
            json.dump(data_serializable, f, indent=2)
        log.info(f"✓ Aggregated data saved to {filepath}")
        
        # Also save summary statistics
        summary = self._compute_summary_statistics()
        summary_file = self.output_dir / "aggregated_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        log.info(f"✓ Summary statistics saved to {summary_file}")
        
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
            "data_points_per_run": [
                len(run["data"]["update_steps"]) for run in self.individual_runs
            ],
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
                "total": float(wall_clock_times.max()),
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
                label=f'Run {int(run_id)}',
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
                label=f'Run {int(run_id)}',
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
        log.info(f"✓ Aggregated plot saved to {filepath}")
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
        log.info(f"✓ Combined plot saved to {filepath}")
        plt.close()
        return filepath
    
    def print_summary(self):
        """Print summary statistics to console."""
        summary = self._compute_summary_statistics()
        
        print("\n" + "="*70)
        print("AGGREGATION SUMMARY")
        print("="*70)
        print(f"Number of runs: {summary['num_runs']}")
        print(f"Total data points: {summary['num_total_data_points']}")
        print(f"Data points per run: {summary['data_points_per_run']}")
        
        print("\nSelf-play Returns:")
        print(f"  Mean: {summary['sp_returns']['mean']:.6f}")
        print(f"  Std:  {summary['sp_returns']['std']:.6f}")
        print(f"  Range: [{summary['sp_returns']['min']:.6f}, {summary['sp_returns']['max']:.6f}]")
        
        print("\nCross-play Returns:")
        print(f"  Mean: {summary['xp_returns']['mean']:.6f}")
        print(f"  Std:  {summary['xp_returns']['std']:.6f}")
        print(f"  Range: [{summary['xp_returns']['min']:.6f}, {summary['xp_returns']['max']:.6f}]")
        
        print("\nWall-clock Times (seconds):")
        print(f"  Mean per point: {summary['wall_clock_times']['mean']:.2f}s")
        print(f"  Std:  {summary['wall_clock_times']['std']:.2f}s")
        print(f"  Total span: {summary['wall_clock_times']['total']:.2f}s")
        print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate BRDiv monitoring data from multiple runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Aggregate from a directory structure
  python aggregate_brdiv_runs.py --input-dir ./brdiv_runs --output-dir ./results
  
  # Aggregate specific directories
  python aggregate_brdiv_runs.py \
    --run-dirs ./run1 ./run2 ./run3 \
    --output-dir ./aggregated
  
  # Custom glob pattern
  python aggregate_brdiv_runs.py \
    --input-dir ./results \
    --output-dir ./aggregated \
    --pattern "**/monitoring/brdiv_monitoring_data.json"
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--input-dir',
        type=str,
        help='Directory to search for monitoring files'
    )
    input_group.add_argument(
        '--run-dirs',
        type=str,
        nargs='+',
        help='Specific run directories to aggregate'
    )
    input_group.add_argument(
        '--files',
        type=str,
        nargs='+',
        help='Specific JSON files to aggregate'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./brdiv_aggregated',
        help='Directory to save aggregated results (default: ./brdiv_aggregated)'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='**/monitoring/brdiv_monitoring_data.json',
        help='Glob pattern for finding monitoring files (default: **/monitoring/brdiv_monitoring_data.json)'
    )
    
    args = parser.parse_args()
    
    # Create aggregator
    aggregator = BRDivAggregator(output_dir=args.output_dir)
    
    # Load data based on input mode
    if args.input_dir:
        log.info(f"Searching for monitoring files in: {args.input_dir}")
        aggregator.load_runs_from_directory(args.input_dir, pattern=args.pattern)
    elif args.run_dirs:
        log.info(f"Loading from {len(args.run_dirs)} specified directories")
        aggregator.load_runs_from_dirs(args.run_dirs)
    elif args.files:
        log.info(f"Loading {len(args.files)} specified files")
        for idx, file_path in enumerate(args.files):
            aggregator.load_run_from_file(file_path, run_id=idx)
    
    if len(aggregator.individual_runs) == 0:
        log.error("No data loaded!")
        sys.exit(1)
    
    # Save and plot
    log.info("Saving aggregated data...")
    aggregator.save_aggregated_data()
    
    log.info("Generating plots...")
    aggregator.plot_aggregated_results()
    aggregator.plot_aggregated_results_combined()
    
    # Print summary
    aggregator.print_summary()
    
    log.info(f"✓ All results saved to: {aggregator.output_dir}")


if __name__ == '__main__':
    main()
