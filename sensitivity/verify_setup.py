#!/usr/bin/env python
"""
Verification script to check if all dependencies and data files are present
for Model 1 training with wandb integration.
"""

import sys
import os
from pathlib import Path

def check_python_packages():
    """Check if all required Python packages are installed."""
    required_packages = [
        'pandas',
        'numpy',
        'sklearn',
        'tensorflow',
        'rdkit',
        'matplotlib',
        'seaborn',
        'wandb'
    ]
    
    print("Checking Python packages...")
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (MISSING)")
            missing.append(package)
    
    return len(missing) == 0, missing


def check_data_files():
    """Check if all required data files exist."""
    required_files = [
        'dataset/GDSC2_cnd_overlap_with_positions.csv',
        'dataset/Cells/Cell1.csv',
        'dataset/Drugs/SMILES/smiles.csv'
    ]
    
    print("\nChecking data files...")
    missing = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
            print(f"  ✓ {file_path} ({file_size:.2f} MB)")
        else:
            print(f"  ✗ {file_path} (MISSING)")
            missing.append(file_path)
    
    return len(missing) == 0, missing


def check_output_dirs():
    """Check if output directories exist, create if needed."""
    dirs = ['results']
    
    print("\nChecking output directories...")
    for dir_name in dirs:
        if os.path.exists(dir_name):
            print(f"  ✓ {dir_name}/")
        else:
            try:
                os.makedirs(dir_name, exist_ok=True)
                print(f"  ✓ {dir_name}/ (created)")
            except Exception as e:
                print(f"  ✗ {dir_name}/ (failed to create: {e})")
                return False
    
    return True


def check_wandb_login():
    """Check if user is logged into wandb."""
    print("\nChecking wandb login...")
    try:
        import wandb
        api = wandb.Api()
        try:
            me = api.viewer()
            print(f"  ✓ Logged in as: {me.entity}")
            return True
        except Exception:
            print(f"  ✗ Not logged in to wandb")
            print(f"    Run: wandb login")
            return False
    except Exception as e:
        print(f"  ✗ Error checking wandb: {e}")
        return False


def main():
    print("=" * 70)
    print("MODEL 1 SETUP VERIFICATION")
    print("=" * 70)
    
    packages_ok, missing_packages = check_python_packages()
    files_ok, missing_files = check_data_files()
    dirs_ok = check_output_dirs()
    wandb_ok = check_wandb_login()
    
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    
    all_ok = packages_ok and files_ok and dirs_ok and wandb_ok
    
    if packages_ok:
        print("✓ Python packages: OK")
    else:
        print(f"✗ Python packages: MISSING - {', '.join(missing_packages)}")
        print("  Fix with: pip install -r requirements.txt")
    
    if files_ok:
        print("✓ Data files: OK")
    else:
        print(f"✗ Data files: MISSING - {', '.join(missing_files)}")
    
    if dirs_ok:
        print("✓ Output directories: OK")
    else:
        print("✗ Output directories: FAILED")
    
    if wandb_ok:
        print("✓ wandb login: OK")
    else:
        print("✗ wandb login: FAILED")
        print("  Fix with: wandb login")
    
    print("=" * 70)
    
    if all_ok:
        print("✓ All checks passed! Ready to train.")
        print("\nTo start training, run:")
        print("  python model_training.py")
        print("\nOr use the script:")
        print("  Windows: train_model1.bat")
        print("  Linux/WSL: ./train_model1.sh")
        return 0
    else:
        print("✗ Some checks failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
