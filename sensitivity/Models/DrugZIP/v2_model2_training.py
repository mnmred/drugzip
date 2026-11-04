# -*- coding: utf-8 -*-
"""
Model training pipeline with cross-validation.
- Model 2: InChIKey Vector + Cell → ln_ic50 (direct vector features)
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import wandb
from wandb.integration.keras import WandbCallback

wandb.login()  # reads the WANDB_API_KEY environment variable or a previous `wandb login`

warnings.filterwarnings('ignore')

np.random.seed(42)
tf.random.set_seed(42)


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

def build_dnn_model(input_dim=128, output_dim=1, dropout_rate=0.2):
    """Residual DNN suited for compressed latent-space inputs."""

    inputs = keras.Input(shape=(input_dim,))

    # 1. Expand before contracting — give the net room to "unfold" the latent space
    x = keras.layers.Dense(512, activation='linear')(inputs)
    x = keras.layers.LayerNormalization()(x)   # LN > BN for latent inputs
    x = keras.layers.Activation('gelu')(x)     # GELU > ReLU for smooth gradients

    # 2. Wide residual blocks — depth without vanishing gradients
    for units in [512, 512, 256, 256, 128]:
        residual = x if x.shape[-1] == units else keras.layers.Dense(units, use_bias=False)(x)
        x = keras.layers.Dense(units, activation='linear')(x)
        x = keras.layers.LayerNormalization()(x)
        x = keras.layers.Activation('gelu')(x)
        x = keras.layers.Dropout(dropout_rate)(x)
        x = keras.layers.Dense(units, activation='linear')(x)
        x = keras.layers.LayerNormalization()(x)
        x = keras.layers.Add()([x, residual])  # skip connection
        x = keras.layers.Activation('gelu')(x)

    # 3. Narrow bottleneck head
    x = keras.layers.Dense(64, activation='gelu')(x)
    x = keras.layers.Dropout(dropout_rate / 2)(x)
    outputs = keras.layers.Dense(output_dim, activation='linear')(x)

    return keras.Model(inputs, outputs)


def build_optimizer():
    """Plain Adam, matching the original training recipe."""
    return keras.optimizers.Adam(learning_rate=1e-3)


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_model2_data():
    """Load and prepare Model 2 data from CSV files."""
    print("\n=== Loading Model 2 Data ===")

    print("Loading GDSC2_cnd_overlap_with_positions.csv...")
    main_data = pd.read_csv('dataset/GDSC2_cnd_overlap_with_positions.csv')
    print(f"Loaded {len(main_data)} samples")

    print("Loading Cell1.csv...")
    cell_data = pd.read_csv('dataset/Cells/Cell1.csv', index_col=0)
    print(f"Cell data shape: {cell_data.shape}")

    print("Loading VAE strings...")
    vae_list = []
    vae_25in1_file = 'dataset/Drugs/25in1-full-b256.csv'
    a1_file = '/path/to/CCSynergy/Sanger_Full/A1.csv'
    with open(vae_25in1_file, 'r') as f:
        for line in f:
            vector = line.strip()
            if vector:
                vae_list.append(vector)
    print(f"Loaded {len(vae_list)} VAE strings")

    model2_data = []
    for idx, row in main_data.iterrows():
        cell_name    = row['CELL_LINE_NAME']
        drug_position = int(row['DRUG_POSITION'])
        ln_ic50      = row['LN_IC50']

        if cell_name in cell_data.columns:
            cell_vector = cell_data[cell_name].values
        else:
            print(f"Warning: Cell {cell_name} not found in Cell1.csv, skipping...")
            continue

        if 0 <= drug_position < len(vae_list):
            vae_string = vae_list[drug_position]
        else:
            print(f"Warning: Drug position {drug_position} out of range, skipping row {idx}..., "
                  f"limit is {len(vae_list) - 1}")
            continue

        model2_data.append({
            'CELL_NAME':     cell_name,
            'DRUG_POSITION': drug_position,
            'VAE_STRING':    vae_string,
            'CELL_VECTOR':   cell_vector,
            'LN_IC50':       ln_ic50,
        })

    model2_df = pd.DataFrame(model2_data)
    print(f"Final model2 data: {len(model2_df)} samples")
    print(f"Number of unique cells: {model2_df['CELL_NAME'].nunique()}")
    print(f"Number of unique drugs: {model2_df['DRUG_POSITION'].nunique()}")
    print(f"Number of unique cell-drug pairs: "
          f"{model2_df.groupby(['CELL_NAME', 'DRUG_POSITION']).size().shape[0]}")

    return model2_df


def prepare_model2_features(model2_data):
    """Prepare features for Model 2 (VAE Vector + Cell)."""
    print("\n=== Preparing Model 2 Features ===")

    vae_vectors = []
    for vae_str in model2_data['VAE_STRING'].values:
        vae_vec = np.array([float(x.strip('"')) for x in vae_str.split(',')[1:]])
        vae_vectors.append(vae_vec)
    vae_vectors = np.array(vae_vectors)

    cell_vectors = np.array(list(model2_data['CELL_VECTOR'].values))
    X = np.hstack([vae_vectors, cell_vectors])
    y = model2_data['LN_IC50'].values

    print(f"VAE vectors shape:  {vae_vectors.shape}")
    print(f"Cell vectors shape: {cell_vectors.shape}")
    print(f"Model 2 feature shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    return X, y


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train_model_with_cv(X, y, model_name, n_splits=10, epochs=500, batch_size=256):
    """Train model with k-fold cross-validation and wandb logging."""
    print(f"\n=== Training {model_name} with {n_splits}-Fold CV ===")

    wandb.init(
        project="Sensitivity-CV1",
        name=model_name,
        config={
            "model_name":    model_name,
            "n_splits":      n_splits,
            "epochs":        epochs,
            "batch_size":    batch_size,
            "input_dim":     X.shape[1],
            "dataset_size":  len(X),
            "architecture":  "residual_dnn_v2",
            "optimizer":     "AdamW+CosineRestarts",
            "dropout_rate":  0.2,
            "weight_decay":  1e-4,
            "clipnorm":      1.0,
        }
    )

    # ── shuffle & hold-out split ──────────────────────────────────────────────
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X, y = X[indices], y[indices]

    split = int(0.8 * len(X))
    X_trainval, X_test = X[:split], X[split:]
    y_trainval, y_test = y[:split], y[split:]

    # ── cross-validation ─────────────────────────────────────────────────────
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_results = {k: [] for k in
                    ['fold', 'train_mse', 'val_mse',
                     'train_mae', 'val_mae', 'train_r2', 'val_r2']}

    trained_models  = []
    best_val_r2     = -np.inf
    best_model      = None          # ← track the actual best model, not just its fold

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_trainval), 1):
        print(f"\n--- Fold {fold}/{n_splits} ---")

        X_train, X_val = X_trainval[train_idx], X_trainval[val_idx]
        y_train, y_val = y_trainval[train_idx], y_trainval[val_idx]


        model = build_dnn_model(input_dim=X_train.shape[1], output_dim=1, dropout_rate=0.2)
        model.compile(
            optimizer=build_optimizer(),
            loss='mse',
            metrics=['mae'],
        )

        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=0,
        )

        wandb_callback = WandbCallback(log_weights=False, log_gradients=False)

        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, wandb_callback],
            verbose=0,
        )

        y_train_pred = model.predict(X_train, verbose=0).flatten()
        y_val_pred   = model.predict(X_val,   verbose=0).flatten()

        train_mse = mean_squared_error(y_train, y_train_pred)
        val_mse   = mean_squared_error(y_val,   y_val_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        val_mae   = mean_absolute_error(y_val,   y_val_pred)
        train_r2  = r2_score(y_train, y_train_pred)
        val_r2    = r2_score(y_val,   y_val_pred)

        print(f"Train MSE: {train_mse:.4f}  Val MSE: {val_mse:.4f}")
        print(f"Train MAE: {train_mae:.4f}  Val MAE: {val_mae:.4f}")
        print(f"Train R²:  {train_r2:.4f}  Val R²:  {val_r2:.4f}")

        for key, val in [('fold', fold),
                         ('train_mse', train_mse), ('val_mse', val_mse),
                         ('train_mae', train_mae), ('val_mae', val_mae),
                         ('train_r2',  train_r2),  ('val_r2',  val_r2)]:
            fold_results[key].append(val)

        wandb.log({
            f"fold_{fold}_train_mse": train_mse, f"fold_{fold}_val_mse": val_mse,
            f"fold_{fold}_train_mae": train_mae, f"fold_{fold}_val_mae": val_mae,
            f"fold_{fold}_train_r2":  train_r2,  f"fold_{fold}_val_r2":  val_r2,
        })

        trained_models.append(model)

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_model  = model                          # ← keep reference
            model.save(f'results/{model_name}_best_model_fold{fold}.h5')
            print(f"  ✓ New best model saved (val R²={val_r2:.4f})")

    # ── final test evaluation using the BEST model ────────────────────────────
    print(f"\n=== Final Test Set Evaluation (best fold, val R²={best_val_r2:.4f}) ===")
    y_test_pred = best_model.predict(X_test, verbose=0).flatten()
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2  = r2_score(y_test, y_test_pred)
    print(f"Test MSE: {test_mse:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Test R²:  {test_r2:.4f}")

    # ── CV summary ────────────────────────────────────────────────────────────
    print(f"\n=== {model_name} Cross-Validation Summary ===")
    mean_val_mse = np.mean(fold_results['val_mse'])
    std_val_mse  = np.std(fold_results['val_mse'])
    mean_val_mae = np.mean(fold_results['val_mae'])
    std_val_mae  = np.std(fold_results['val_mae'])
    mean_val_r2  = np.mean(fold_results['val_r2'])
    std_val_r2   = np.std(fold_results['val_r2'])

    print(f"Val MSE: {mean_val_mse:.4f} ± {std_val_mse:.4f}")
    print(f"Val MAE: {mean_val_mae:.4f} ± {std_val_mae:.4f}")
    print(f"Val R²:  {mean_val_r2:.4f} ± {std_val_r2:.4f}")

    wandb.log({
        "mean_val_mse": mean_val_mse, "std_val_mse": std_val_mse,
        "mean_val_mae": mean_val_mae, "std_val_mae": std_val_mae,
        "mean_val_r2":  mean_val_r2,  "std_val_r2":  std_val_r2,
        "test_mse": test_mse, "test_mae": test_mae, "test_r2": test_r2,
    })

    results_df = pd.DataFrame(fold_results)
    results_df.to_csv(f'results/{model_name}_cv_results.csv', index=False)
    wandb.save(f'results/{model_name}_cv_results.csv')

    wandb.finish()

    return results_df, trained_models


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train Model 2 with wandb integration')
    parser.add_argument('--project',    type=str, default='Sensitivity-CV1')
    parser.add_argument('--entity',     type=str, default=None)
    parser.add_argument('--n_splits',   type=int, default=10)
    parser.add_argument('--epochs',     type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()

    print("=" * 80)
    print("MODEL 2: VAE + Cell → ln_ic50")
    print("=" * 80)
    print(f"wandb project : {args.project}")
    print(f"n_splits      : {args.n_splits}")
    print(f"epochs        : {args.epochs}")
    print(f"batch_size    : {args.batch_size}")
    print("=" * 80)

    model2_data       = load_model2_data()
    X, y              = prepare_model2_features(model2_data)
    results_df, models = train_model_with_cv(
        X, y,
        model_name  = "Model2_VAE_Cell",
        n_splits    = args.n_splits,
        epochs      = args.epochs,
        batch_size  = args.batch_size,
    )

    print("\n" + "=" * 80)
    print("Model 2 training complete!")
    print("=" * 80)