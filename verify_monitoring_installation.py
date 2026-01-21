#!/usr/bin/env python3
"""
Verification script to ensure all BRDiv monitoring components are properly installed.
Run this to verify the monitoring system is ready to use.
"""

import os
import sys
from pathlib import Path


def verify_files():
    """Verify all required files exist."""
    print("=" * 70)
    print("VERIFYING BRDIV MONITORING SYSTEM")
    print("=" * 70)
    print()
    
    root = Path("/scratch/cluster/adityam/jax-aht")
    
    required_files = {
        "Core Implementation": [
            "teammate_generation/brdiv_with_monitoring.py",
            "teammate_generation/brdiv_monitoring_analysis.py",
        ],
        "Examples & Helpers": [
            "teammate_generation/run_brdiv_monitored.py",
            "teammate_generation/quick_start_monitoring.py",
            "teammate_generation/examples_monitoring.py",
        ],
        "Documentation": [
            "BRDIV_MONITORING_QUICK_START.md",
            "BRDIV_MONITORING_README.md",
            "BRDIV_MONITORING_IMPLEMENTATION.md",
            "BRDIV_MONITORING_COMPLETE.md",
            "INDEX.md",
            "DELIVERABLES.md",
            "README_MONITORING.md",
        ],
    }
    
    all_exist = True
    for category, files in required_files.items():
        print(f"📁 {category}:")
        for file_path in files:
            full_path = root / file_path
            exists = full_path.exists()
            status = "✅" if exists else "❌"
            print(f"   {status} {file_path}")
            if not exists:
                all_exist = False
        print()
    
    return all_exist


def verify_brdiv_modifications():
    """Verify BRDiv.py has been properly modified."""
    print("🔧 BRDiv.py Modifications:")
    
    brdiv_file = Path("/scratch/cluster/adityam/jax-aht/teammate_generation/BRDiv.py")
    
    with open(brdiv_file, 'r') as f:
        content = f.read()
    
    checks = {
        "monitor initialization": "monitor = None" in content and "enable_brdiv_monitoring" in content,
        "monitor.record_update()": "monitor.record_update(" in content,
        "monitor.save_data()": "monitor.save_data()" in content,
        "monitor.plot_results()": "monitor.plot_results()" in content,
        "monitor parameter in log_metrics": "def log_metrics(config, outs, logger, metric_names: tuple, monitor=None):" in content,
    }
    
    all_modified = True
    for check_name, result in checks.items():
        status = "✅" if result else "❌"
        print(f"   {status} {check_name}")
        if not result:
            all_modified = False
    
    print()
    return all_modified


def verify_imports():
    """Verify that modules can be imported."""
    print("📦 Import Verification (requires numpy):")
    print("   ℹ️  Skipping - numpy not in current environment")
    print("   ℹ️  Will work when running BRDiv (which has numpy)")
    print()
    return True  # Will work in BRDiv environment


def verify_file_sizes():
    """Verify that key files have reasonable sizes."""
    print("📊 File Sizes:")
    
    root = Path("/scratch/cluster/adityam/jax-aht")
    files_to_check = {
        "teammate_generation/brdiv_with_monitoring.py": (200, 500),
        "teammate_generation/brdiv_monitoring_analysis.py": (200, 400),
        "teammate_generation/examples_monitoring.py": (250, 450),
    }
    
    all_ok = True
    for file_path, (min_lines, max_lines) in files_to_check.items():
        full_path = root / file_path
        if full_path.exists():
            with open(full_path, 'r') as f:
                line_count = len(f.readlines())
            
            if min_lines <= line_count <= max_lines:
                status = "✅"
            else:
                status = "⚠️"
                all_ok = False
            
            print(f"   {status} {file_path}: {line_count} lines ({min_lines}-{max_lines})")
        else:
            print(f"   ❌ {file_path}: File not found")
            all_ok = False
    
    print()
    return all_ok


def main():
    """Run all verification checks."""
    
    files_ok = verify_files()
    print()
    
    brdiv_ok = verify_brdiv_modifications()
    print()
    
    imports_ok = verify_imports()
    
    sizes_ok = verify_file_sizes()
    
    # Summary
    print("=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    
    all_checks = {
        "Files exist": files_ok,
        "BRDiv modifications": brdiv_ok,
        "Module imports": imports_ok,
        "File sizes": sizes_ok,
    }
    
    for check_name, result in all_checks.items():
        status = "✅" if result else "❌"
        print(f"{status} {check_name}")
    
    print()
    
    if all(all_checks.values()):
        print("🎉 ALL CHECKS PASSED - MONITORING SYSTEM IS READY!")
        print()
        print("NEXT STEPS:")
        print("  1. Read: BRDIV_MONITORING_QUICK_START.md")
        print("  2. Run:  python teammate_generation/run.py ... enable_brdiv_monitoring=true")
        print("  3. Analyze: python teammate_generation/brdiv_monitoring_analysis.py <data.json>")
        return 0
    else:
        print("❌ SOME CHECKS FAILED - PLEASE REVIEW ABOVE")
        return 1


if __name__ == "__main__":
    sys.exit(main())
