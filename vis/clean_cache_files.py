import os
import argparse

CACHE_FILENAME = "cached_summary_metrics.pkl"
HELDOUT_CURVES_CACHE_FILENAME = "cached_heldout_curves.pkl"

# Files to look for
TARGET_FILES = [
    CACHE_FILENAME,
    HELDOUT_CURVES_CACHE_FILENAME,
    CACHE_FILENAME.replace('.pkl', '_renormalized.pkl'),
    HELDOUT_CURVES_CACHE_FILENAME.replace('.pkl', '_renormalized.pkl')
]

def find_cache_files(results_dir):
    """Find all cache files in the results directory structure."""
    cache_files = []
    
    # Define task directories
    task_dirs = []
    
    # Add LBF directory
    lbf_dir = os.path.join(results_dir, "lbf")
    if os.path.exists(lbf_dir) and os.path.isdir(lbf_dir):
        task_dirs.append(lbf_dir)
    
    # Add Overcooked subdirectories
    overcooked_dir = os.path.join(results_dir, "overcooked-v1")
    if os.path.exists(overcooked_dir) and os.path.isdir(overcooked_dir):
        for overcooked_task in os.listdir(overcooked_dir):
            task_path = os.path.join(overcooked_dir, overcooked_task)
            if os.path.isdir(task_path):
                task_dirs.append(task_path)
    
    # For each task directory, find all heldout_eval_metrics directories
    for task_dir in task_dirs:
        print(f"Scanning task directory: {task_dir}")
        for method_dir in os.listdir(task_dir):
            method_path = os.path.join(task_dir, method_dir)
            
            # Skip if not a directory or is a special file
            if not os.path.isdir(method_path) or method_dir.startswith('.'):
                continue
                
            # For each method directory, check each version subdirectory
            for version_dir in os.listdir(method_path):
                version_path = os.path.join(method_path, version_dir)
                
                if not os.path.isdir(version_path):
                    continue
                    
                # For each version directory, check each date-based run directory
                for run_dir in os.listdir(version_path):
                    run_path = os.path.join(version_path, run_dir)
                    
                    if not os.path.isdir(run_path):
                        continue
                        
                    # Check for heldout_eval_metrics directory
                    eval_metrics_dir = os.path.join(run_path, "heldout_eval_metrics")
                    if os.path.exists(eval_metrics_dir) and os.path.isdir(eval_metrics_dir):
                        # Look for cache files
                        for target_file in TARGET_FILES:
                            cache_file = os.path.join(eval_metrics_dir, target_file)
                            if os.path.exists(cache_file):
                                cache_files.append(cache_file)
    
    return cache_files

def delete_files(file_list):
    """Delete all files in the given list."""
    for file_path in file_list:
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Clean cache files from results directories')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Path to the results directory (default: results)')
    parser.add_argument('--force', action='store_true',
                        help='Delete files without confirmation')
    args = parser.parse_args()
    
    # Find cache files
    cache_files = find_cache_files(args.results_dir)
    
    if not cache_files:
        print("No cache files found.")
        return
    
    # Display found files
    print(f"\nFound {len(cache_files)} cache files:")
    for i, file_path in enumerate(cache_files, 1):
        print(f"{i}. {file_path}")
    
    # Ask for confirmation
    if not args.force:
        confirmation = input("\nDelete these files? (y/n): ")
        if confirmation.lower() != 'y':
            print("Operation cancelled.")
            return
    
    # Delete files
    print("\nDeleting files...")
    delete_files(cache_files)
    print(f"\nDeleted {len(cache_files)} cache files.")

if __name__ == "__main__":
    main() 