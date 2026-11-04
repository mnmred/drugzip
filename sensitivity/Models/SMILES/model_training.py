# -*- coding: utf-8 -*-
"""
Model training pipeline with cross-validation.
- Model 1: SMILES + Cell → ln_ic50 (using RDKit for SMILES featurization)
- Model 2: InChIKey Vector + Cell → ln_ic50 (direct vector features)
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import KFold, cross_validate, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from rdkit import Chem
from rdkit.Chem import AllChem
import warnings
import pickle
import shap
import wandb
from wandb.integration.keras import WandbCallback
# from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

wandb.login()  # reads the WANDB_API_KEY environment variable or a previous `wandb login`


warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


def smiles_to_fingerprint(smiles_str, radius=2, nbits=1024):
    """Convert SMILES string to Morgan fingerprint."""
    mol = Chem.MolFromSmiles(smiles_str)
    if mol is None:
        return np.zeros(nbits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    return np.array(fp, dtype=np.float32)


def build_dnn_model(input_dim, output_dim=1, hidden_layers=[256, 128, 64], dropout_rate=0.3):
    """Build a simple DNN model."""
    model = keras.Sequential()
    
    # Input layer
    model.add(keras.layers.Input(shape=(input_dim,)))
    
    # Hidden layers
    for units in hidden_layers:
        model.add(keras.layers.Dense(units, activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(dropout_rate))
    
    # Output layer
    model.add(keras.layers.Dense(output_dim, activation='linear'))
    
    return model


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

def load_model1_ohe_data(drug_col='DRUG_POSITION'):
    """Load Model 1 data for OHE. No SMILES are read, so no drug is dropped.

    Only rows whose cell line is missing from Cell1.csv are skipped,
    because those rows have no cell features.
    """
    print("\n=== Loading Model 1 Data (OHE) ===")
    main_data = pd.read_csv('dataset/GDSC2_cnd_overlap_with_positions.csv')
    cell_data = pd.read_csv('dataset/Cells/Cell1.csv', index_col=0)
    print(f"Loaded {len(main_data)} samples, cell data shape: {cell_data.shape}")

    if main_data[drug_col].isna().any():
        raise ValueError(f"{main_data[drug_col].isna().sum()} rows have no {drug_col}; "
                         "pass another drug identifier column as drug_col")

    has_cell = main_data['CELL_LINE_NAME'].isin(cell_data.columns)
    if not has_cell.all():
        print(f"Warning: skipping {(~has_cell).sum()} rows whose cell line is not in Cell1.csv")
    main_data = main_data[has_cell].reset_index(drop=True)

    model1_df = pd.DataFrame({
        'CELL_NAME': main_data['CELL_LINE_NAME'],
        'DRUG_ID': main_data[drug_col],
        'CELL_VECTOR': [cell_data[c].values for c in main_data['CELL_LINE_NAME']],
        'LN_IC50': main_data['LN_IC50'],
    })
    print(f"Final model1 OHE data: {len(model1_df)} samples, "
          f"{model1_df['CELL_NAME'].nunique()} cells, {model1_df['DRUG_ID'].nunique()} drugs")
    return model1_df


def prepare_model1_ohe_features(model1_data):
    """One-hot encode the drug (one column per unique drug) and append the cell vector."""
    print("\n=== Preparing Model 1 OHE Features ===")
    # np.unique sorts the drug IDs, so the column order is the same every run;
    # drug_cols[i] is the OHE column of row i
    drug_ids, drug_cols = np.unique(model1_data['DRUG_ID'].values, return_inverse=True)
    n_drugs = len(drug_ids)

    ohe = np.zeros((len(model1_data), n_drugs), dtype=np.float32)
    ohe[np.arange(len(model1_data)), drug_cols] = 1.0

    cell_vectors = np.stack(model1_data['CELL_VECTOR'].values)
    X = np.hstack([ohe, cell_vectors])
    y = model1_data['LN_IC50'].values
    print(f"OHE size: {n_drugs} drugs, feature shape: {X.shape}")
    return X, y, n_drugs


def train_model_with_cv(
    X, y, model_name, smiles_feature_count,
    n_splits=10, epochs=500, batch_size=256
):
    """Train model with k-fold cross-validation and wandb logging."""
    print(f"\n=== Training {model_name} with {n_splits}-Fold CV ===")
    
    # Initialize wandb run
    wandb.init(
        project="Sensitivity-Paper",
        name=model_name,
        config={
            "model_name": model_name,
            "n_splits": n_splits,
            "epochs": epochs,
            "batch_size": batch_size,
            "input_dim": X.shape[1],
            "dataset_size": len(X),
            "hidden_layers": [1024, 528, 256]
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


    for fold, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
        print(f"\n--- Fold {fold}/{n_splits} ---")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Build model
        input_dim = X_train.shape[1]
        hidden_layers_model_1 = [1024, 528, 256] 
        model = build_dnn_model(input_dim, output_dim=1, hidden_layers=hidden_layers_model_1)
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss='mse',
            metrics=['mae']
        )
        
        # Early stopping callback
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=0
        )
        
        # wandb callback
        wandb_callback = WandbCallback(
            log_weights=False,
            log_gradients=False
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, wandb_callback],
            verbose=0
        )
        
        # Predictions
        y_train_pred = model.predict(X_train, verbose=0).flatten()
        y_val_pred = model.predict(X_val, verbose=0).flatten()
        
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
            model.save(f'results/{model_name}_best_model_fold{fold}.h5')
    
    y_test_pred = model.predict(test_data, verbose=0).flatten()
    test_mse = mean_squared_error(test_labels, y_test_pred)
    test_mae = mean_absolute_error(test_labels, y_test_pred)    
    test_r2 = r2_score(test_labels, y_test_pred)
    print(f"\n=== Final Test Set Evaluation ===")
    print(f"Test MSE: {test_mse:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Test R²: {test_r2:.4f}")

    print("\n--- Running SHAP Analysis ---")
    if not 0 < smiles_feature_count < X.shape[1]:
        raise ValueError(
            f"Expected SMILES fingerprint and cell features, got "
            f"{smiles_feature_count} fingerprint features out of {X.shape[1]} "
            "total features"
        )

    rng = np.random.default_rng(42)
    background_idx = rng.choice(
        train_data.shape[0], size=min(50, train_data.shape[0]), replace=False
    )
    shap_background = train_data[background_idx]
    explanation_idx = rng.choice(
        test_data.shape[0], size=min(100, test_data.shape[0]), replace=False
    )
    shap_test_data = test_data[explanation_idx]

    explainer = shap.GradientExplainer(model, shap_background)
    shap_values = explainer.shap_values(shap_test_data, nsamples=100)
    if isinstance(shap_values, list):
        if len(shap_values) != 1:
            raise ValueError(
                f"Expected one regression output, got {len(shap_values)} SHAP outputs"
            )
        shap_values = shap_values[0]

    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3 and shap_values.shape[-1] == 1:
        shap_values = shap_values[..., 0]
    if shap_values.shape != shap_test_data.shape:
        raise ValueError(
            f"SHAP values have shape {shap_values.shape}; expected "
            f"{shap_test_data.shape}"
        )

    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    smiles_importance = np.sum(mean_abs_shap[:smiles_feature_count])
    cell_importance = np.sum(mean_abs_shap[smiles_feature_count:])
    total_importance = smiles_importance + cell_importance
    if total_importance == 0:
        smiles_percentage = 0.0
        cell_percentage = 0.0
    else:
        smiles_percentage = smiles_importance / total_importance * 100
        cell_percentage = cell_importance / total_importance * 100

    print("\n=======================================================")
    print("SHAP FEATURE IMPORTANCE BUCKETS")
    print("=======================================================")
    print(
        f"SMILES fingerprint features (0-{smiles_feature_count - 1}): "
        f"{smiles_percentage:.2f}% of total importance"
    )
    print(
        f"Cell features ({smiles_feature_count}-{X.shape[1] - 1}): "
        f"{cell_percentage:.2f}% of total importance"
    )
    print("=======================================================\n")

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
    parser = argparse.ArgumentParser(description='Train Model 1 with wandb integration')
    parser.add_argument('--project', type=str, default='Sensitivity-Paper', help='Weights & Biases project name')
    parser.add_argument('--entity', type=str, default=None, help='Weights & Biases entity (username or team name)')
    parser.add_argument('--n_splits', type=int, default=10, help='Number of CV folds')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--nbits', type=int, default=1024, help='Number of bits for Morgan fingerprints')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("MF1024 + Cell → ln_ic50 (Morgan Fingerprints)")
    print("=" * 80)
    print(f"wandb project: {args.project}")
    print(f"n_splits: {args.n_splits}")
    print(f"epochs: {args.epochs}")
    print(f"batch_size: {args.batch_size}")
    print("=" * 80)
    
    # Load data
    model1_data = load_model1_data()
    
    # # Prepare features
    X, y, smiles_feat_dim = prepare_model1_features(model1_data, nbits=args.nbits)

    # For OHE, we can use the same function but without SMILES featurization
    # model1_data = load_model1_ohe_data()                                             # OHE
    # X, y, smiles_feat_dim = prepare_model1_ohe_features(model1_data)   
    
    # Train with cross-validation
    results_df, trained_models = train_model_with_cv(
        X, y, 
        model_name="MF1024_cv1",   # was "Model1_SMILES_Cell"
        smiles_feature_count=smiles_feat_dim,
        n_splits=args.n_splits,
        epochs=args.epochs,
        batch_size=args.batch_size
        )
    
    # For OHE
    # results_df, trained_models = train_model_with_cv(
    #     X, y,
    #     model_name="Model1_OHE_CV1_Cell",   # was "Model1_SMILES_Cell"
    #     smiles_feature_count=smiles_feat_dim,
    #     n_splits=args.n_splits,
    #     epochs=args.epochs,
    #     batch_size=args.batch_size
    #     )

    print("\n" + "=" * 80)
    print("Model 1 training complete!")
    print("=" * 80)

