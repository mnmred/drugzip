# -*- coding: utf-8 -*-
"""
Model training pipeline with cross-validation.
- Model 1: SMILES + Cell → ln_ic50 (using RDKit for SMILES featurization)
- Model 2: InChIKey Vector + Cell → ln_ic50 (direct vector features)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_validate, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import ElasticNet
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import AllChem
import warnings
import pickle
import wandb

wandb.login()  # reads the WANDB_API_KEY environment variable or a previous `wandb login`


warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)



def smiles_to_fingerprint(smiles_str, radius=2, nbits=1024):
    """Convert SMILES string to Morgan fingerprint."""
    mol = Chem.MolFromSmiles(smiles_str)
    if mol is None:
        return np.zeros(nbits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    return np.array(fp, dtype=np.float32)


def build_elastic_net_model(alpha=1.0, l1_ratio=0.5, random_state=42):
    """Build an Elastic Net model."""
    return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state, max_iter=5000)



def load_model1_data():
    """Load and prepare Model 1 data from CSV files.
    
    Loads:
    - GDSC2_cnd_overlap_with_positions.csv for cell names, drug positions, and ln_ic50
    - Cell1.csv for cell vectors
    - smiles_no_key_duplicates.csv for SMILES strings indexed by drug position
    """
    print("\n=== Loading Model 1 Data ===")
    
    # Load main data
    print("Loading GDSC2_cnd_overlap_with_positions.csv...")
    main_data = pd.read_csv('dataset/GDSC2_cnd_overlap_with_positions.csv')
    print(f"Loaded {len(main_data)} samples")
    
    # Load cell vectors
    print("Loading Cell1.csv...")
    cell_data = pd.read_csv('dataset/Cells/Cell1.csv', index_col=0)
    print(f"Cell data shape: {cell_data.shape}")
    
    # Load SMILES - read as plain list (no header), but only the second column
    print("Loading SMILES...")
    smiles_list = []
    with open('dataset/Drugs/SMILES/smiles_no_key_duplicates.csv', 'r') as f:
        for line in f:
            smiles = line.strip().split(',')[1]  # Get the second column (SMILES)
            if smiles:
                smiles_list.append(smiles)
            else: 
                print("Warning: Empty SMILES string found, skipping... For key: ", line.strip().split(',')[0])
                smiles_list.append(None)  # Append None for empty SMILES to maintain indexing
    print(f"Loaded {len(smiles_list)} SMILES strings")
    
    # Build model1_data dataframe
    model1_data = []
    
    for idx, row in main_data.iterrows():
        cell_name = row['CELL_LINE_NAME']
        drug_position = int(row['DRUG_POSITION'])
        ln_ic50 = row['LN_IC50']
        
        # Get cell vector (column from Cell1.csv)
        if cell_name in cell_data.columns:
            cell_vector = cell_data[cell_name].values
        else:
            print(f"Warning: Cell {cell_name} not found in Cell1.csv, skipping row {idx}...")
            continue
        
        # Get SMILES (0-indexed from 0-based drug position)
        if drug_position >= 0 and drug_position < len(smiles_list):
            smiles = smiles_list[drug_position]
            if smiles is None:
                print(f"Warning: No SMILES found for drug position {drug_position}, skipping row {idx}...")
                continue
        else:
            print(f"Warning: Drug position {drug_position} out of range, skipping row {idx}...")
            continue
        
        model1_data.append({
            'CELL_NAME': cell_name,
            'DRUG_POSITION': drug_position,
            'SMILES': smiles,
            'CELL_VECTOR': cell_vector,
            'LN_IC50': ln_ic50
        })
        
        # if (idx + 1) % 500 == 0:
        #     print(f"  Processed {idx + 1}/{len(main_data)} samples")
    
    model1_df = pd.DataFrame(model1_data)
    print(f"Final model1 data: {len(model1_df)} samples")
    print(f"Number of unique cells: {model1_df['CELL_NAME'].nunique()}")
    print(f"Number of unique drugs: {model1_df['DRUG_POSITION'].nunique()}")
    print(f"Number of unique cell-drug pairs: {model1_df.groupby(['CELL_NAME', 'DRUG_POSITION']).size().shape[0]}")
    
    return model1_df


def prepare_model1_features(model1_data, nbits=1024):
    """Prepare features for Model 1 (SMILES + Cell)."""
    print("\n=== Preparing Model 1 Features ===")
    print("Converting SMILES to Morgan fingerprints...")
    
    # Convert SMILES to fingerprints
    fingerprints = []
    for idx, smiles in enumerate(model1_data['SMILES']):
        if idx % 100 == 0:
            print(f"  Processed {idx}/{len(model1_data)} SMILES")
        fp = smiles_to_fingerprint(smiles, nbits=nbits)
        fingerprints.append(fp)
    
    fingerprints = np.array(fingerprints)
    
    # Combine with cell encoding (convert from list to numpy array)
    cell_vectors = np.array([cv for cv in model1_data['CELL_VECTOR'].values])
    X = np.hstack([fingerprints, cell_vectors])
    y = model1_data['LN_IC50'].values
    
    print(f"Model 1 Feature shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    return X, y, fingerprints.shape[1]


def prepare_model2_features(model2_data):
    """Prepare features for Model 2 (InChIKey Vector + Cell)."""
    print("\n=== Preparing Model 2 Features ===")
    
    # Load InChIKey vectors
    inchikey_vectors = np.load('dataset/inchikey_vectors.npy')
    
    # Combine with cell encoding
    cell_encoded = model2_data['CELL_ENCODED'].values.reshape(-1, 1)
    X = np.hstack([inchikey_vectors, cell_encoded])
    y = model2_data['LN_IC50'].values
    
    print(f"Model 2 Feature shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    return X, y


def train_model_with_cv(X, y, model_name, n_splits=10, alpha=1.0, l1_ratio=0.5):
    """Train Elastic Net model with k-fold cross-validation and wandb logging."""
    print(f"\n=== Training {model_name} with {n_splits}-Fold CV ===")
    
    # Initialize wandb run
    wandb.init(
        project="Sensitivity-EN-CV1",
        name=model_name,
        config={
            "model_name": model_name,
            "n_splits": n_splits,
            "alpha": alpha,
            "l1_ratio": l1_ratio,
            "input_dim": X.shape[1],
            "dataset_size": len(X)
        }
    )
    
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_results = {
        'fold': [],
        'train_mse': [],
        'val_mse': [],
        'train_mae': [],
        'val_mae': [],
        'train_r2': [],
        'val_r2': []
    }
    
    trained_models = []
    best_val_r2 = -np.inf

    # let's shuffle the data before splitting to ensure randomness (though KFold with shuffle=True should handle this)
    # but we will do it here to ensure the hold-out test set is also randomized
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    
    # split data into train and test (hold out 20% for final evaluation)
    train_data = X[:int(0.8 * len(X))]
    train_labels = y[:int(0.8 * len(y))]
    test_data = X[int(0.8 * len(X)):]
    test_labels = y[int(0.8 * len(y)):]

    X = train_data
    y = train_labels

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    test_data_scaled = scaler.transform(test_data)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_scaled), 1):
        print(f"\n--- Fold {fold}/{n_splits} ---")
        
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Build and train model
        model = build_elastic_net_model(alpha=alpha, l1_ratio=l1_ratio)
        model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        
        # Metrics
        train_mse = mean_squared_error(y_train, y_train_pred)
        val_mse = mean_squared_error(y_val, y_val_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        val_mae = mean_absolute_error(y_val, y_val_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        
        print(f"Train MSE: {train_mse:.4f}, Val MSE: {val_mse:.4f}")
        print(f"Train MAE: {train_mae:.4f}, Val MAE: {val_mae:.4f}")
        print(f"Train R²: {train_r2:.4f}, Val R²: {val_r2:.4f}")
        
        fold_results['fold'].append(fold)
        fold_results['train_mse'].append(train_mse)
        fold_results['val_mse'].append(val_mse)
        fold_results['train_mae'].append(train_mae)
        fold_results['val_mae'].append(val_mae)
        fold_results['train_r2'].append(train_r2)
        fold_results['val_r2'].append(val_r2)
        
        # Log fold results to wandb
        wandb.log({
            f"fold_{fold}_train_mse": train_mse,
            f"fold_{fold}_val_mse": val_mse,
            f"fold_{fold}_train_mae": train_mae,
            f"fold_{fold}_val_mae": val_mae,
            f"fold_{fold}_train_r2": train_r2,
            f"fold_{fold}_val_r2": val_r2
        })
        
        trained_models.append(model)
        
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            # Save best model
            pickle.dump(model, open(f'results/{model_name}_best_model_fold{fold}.pkl', 'wb'))
    
    y_test_pred = model.predict(test_data_scaled)
    test_mse = mean_squared_error(test_labels, y_test_pred)
    test_mae = mean_absolute_error(test_labels, y_test_pred)    
    test_r2 = r2_score(test_labels, y_test_pred)
    print(f"\n=== Final Test Set Evaluation ===")
    print(f"Test MSE: {test_mse:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Test R²: {test_r2:.4f}")

    # Summary statistics
    print(f"\n=== {model_name} Cross-Validation Summary ===")
    mean_val_mse = np.mean(fold_results['val_mse'])
    std_val_mse = np.std(fold_results['val_mse'])
    mean_val_mae = np.mean(fold_results['val_mae'])
    std_val_mae = np.std(fold_results['val_mae'])
    mean_val_r2 = np.mean(fold_results['val_r2'])
    std_val_r2 = np.std(fold_results['val_r2'])
    
    print(f"Val MSE: {mean_val_mse:.4f} ± {std_val_mse:.4f}")
    print(f"Val MAE: {mean_val_mae:.4f} ± {std_val_mae:.4f}")
    print(f"Val R²:  {mean_val_r2:.4f} ± {std_val_r2:.4f}")
    
    # Log summary to wandb
    wandb.log({
        "mean_val_mse": mean_val_mse,
        "std_val_mse": std_val_mse,
        "mean_val_mae": mean_val_mae,
        "std_val_mae": std_val_mae,
        "mean_val_r2": mean_val_r2,
        "std_val_r2": std_val_r2,
        "test_mse": test_mse,
        "test_mae": test_mae,   
        "test_r2": test_r2
    })
    
    # Save results
    results_df = pd.DataFrame(fold_results)
    results_df.to_csv(f'results/{model_name}_cv_results.csv', index=False)
    
    # Save results CSV to wandb
    wandb.save(f'results/{model_name}_cv_results.csv')
    
    wandb.finish()
    
    return results_df, trained_models





if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Model 1 with Elastic Net and wandb integration')
    parser.add_argument('--project', type=str, default='Sensitivity-cv1', help='Weights & Biases project name')
    parser.add_argument('--entity', type=str, default=None, help='Weights & Biases entity (username or team name)')
    parser.add_argument('--n_splits', type=int, default=10, help='Number of CV folds')
    parser.add_argument('--alpha', type=float, default=1.0, help='Regularization strength')
    parser.add_argument('--l1_ratio', type=float, default=0.5, help='L1 ratio (0=Ridge, 1=Lasso)')
    parser.add_argument('--nbits', type=int, default=1024, help='Number of bits for Morgan fingerprints')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("MODEL 1: SMILES + Cell → ln_ic50 (Elastic Net)")
    print("=" * 80)
    print(f"wandb project: {args.project}")
    print(f"n_splits: {args.n_splits}")
    print(f"alpha: {args.alpha}")
    print(f"l1_ratio: {args.l1_ratio}")
    print("=" * 80)
    
    # Load data
    model1_data = load_model1_data()
    
    # Prepare features
    X, y, smiles_feat_dim = prepare_model1_features(model1_data, nbits=args.nbits)
    
    # Train with cross-validation
    results_df, trained_models = train_model_with_cv(
        X, y, 
        model_name="Model1_SMILES_Cell",
        n_splits=args.n_splits,
        alpha=args.alpha,
        l1_ratio=args.l1_ratio
        )
    
    print("\n" + "=" * 80)
    print("Model 1 training complete!")
    print("=" * 80)


