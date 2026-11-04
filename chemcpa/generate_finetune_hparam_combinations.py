#!/usr/bin/env python3
"""
Generate random hyperparameter combinations for finetune hyperparameter sweep.
Based on config_sciplex_finetune_num_genes.yaml random section.
This replaces SEML's random sampling without requiring MongoDB.
"""
import random
import numpy as np
import json
from pathlib import Path

def loguniform(low, high, size=1):
    """Sample from log-uniform distribution"""
    return np.exp(np.random.uniform(np.log(low), np.log(high), size))

def generate_finetune_hyperparameter_combinations(num_samples=10, seed=42):
    """Generate random hyperparameter combinations based on SEML config_sciplex_finetune_num_genes.yaml"""
    random.seed(seed)
    np.random.seed(seed)
    
    combinations = []
    
    for i in range(num_samples):
        hparams = {
            # Choice parameters
            'batch_size': 32,
            'adversary_width': random.choice([64, 128, 256]),
            'adversary_depth': random.choice([2, 3, 4]),
            'adversary_steps': random.choice([2, 3]),
            
            # Log-uniform parameters
            'autoencoder_lr': float(loguniform(1e-4, 1e-2, 1)[0]),
            'autoencoder_wd': float(loguniform(1e-8, 1e-5, 1)[0]),
            'adversary_lr': float(loguniform(5e-5, 1e-2, 1)[0]),
            'adversary_wd': float(loguniform(1e-8, 1e-3, 1)[0]),
            'reg_adversary': float(loguniform(1, 40, 1)[0]),
            'reg_adversary_cov': float(loguniform(3, 50, 1)[0]),
            'penalty_adversary': float(loguniform(0.05, 2, 1)[0]),
            'dosers_lr': float(loguniform(1e-4, 1e-2, 1)[0]),
            'dosers_wd': float(loguniform(1e-8, 1e-5, 1)[0]),
        }
        
        combinations.append(hparams)
    
    return combinations

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate finetune hyperparameter combinations')
    parser.add_argument('--num-samples', type=int, default=10, help='Number of random samples (default: 10, matching config)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42, matching config)')
    parser.add_argument('--output', type=str, default='finetune_hparam_combinations.json', help='Output JSON file')
    args = parser.parse_args()
    
    combinations = generate_finetune_hyperparameter_combinations(args.num_samples, args.seed)
    
    # Save to JSON file
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(combinations, f, indent=2)
    
    print(f"Generated {len(combinations)} hyperparameter combinations")
    print(f"Saved to: {output_path}")
    print(f"\nFirst combination:")
    for key, value in combinations[0].items():
        print(f"  {key}: {value}")
