import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
import wandb
from wandb.integration.keras import WandbMetricsLogger

# Uses WANDB_API_KEY from the environment when present
if os.environ.get("WANDB_API_KEY"):
    wandb.login(key=os.environ["WANDB_API_KEY"])


def _ensure_trailing_sep(path: str) -> str:
    return path if path.endswith(("/", "\\")) else path + os.sep


# Usually in Matrix MxN, M is number of samples, N is number of features. This adapts our 
# signature matrix accordingly.
def _max_samples_from_env() -> int | None:
    raw = os.environ.get("DRUGZIP_MAX_SAMPLES")
    return int(raw) if raw else None


def build_input(directory_path):
    directory_path = _ensure_trailing_sep(directory_path)
    max_samples = _max_samples_from_env()
    npy_path = directory_path + "input_matrix.npy"

    # Check if input_matrix.npy exists, load if it does, otherwise build it
    if os.path.exists(npy_path):
        print("Loading input matrix from file...")
        if max_samples is not None:
            mm = np.load(npy_path, mmap_mode="r")
            n = min(max_samples, mm.shape[0])
            print(f"Smoke test: using first {n} / {mm.shape[0]} rows")
            # copy: mmap views are read-only and break in-place shuffle
            return np.array(mm[:n], dtype=np.float64, copy=True)
        return np.load(npy_path)
    else:
        print("Building input matrix...")

    num_drugs = 1230665
    matrix = np.zeros((num_drugs, 25 * 128))
    
    FeatureName = ['A1','A2','A3','A4','A5','B1','B2','B3','B4','B5',
                   'C1','C2','C3','C4','C5','D1','D2','D3','D4','D5',
                   'E1','E2','E3','E4','E5']

    for i, signature in enumerate(FeatureName):
        file_path = f"{directory_path}{signature}.csv"
        print(f"Loading {signature}...")
        df = pd.read_csv(file_path, usecols=range(1, 129), dtype=np.float64, engine='c')
        
        # Vectorized
        matrix[:, i*128 : (i+1)*128] = df.values
        
        # Free up memory immediately
        del df 

    np.save(npy_path, matrix)
    if max_samples is not None:
        n = min(max_samples, matrix.shape[0])
        print(f"Smoke test: using first {n} / {matrix.shape[0]} rows")
        return matrix[:n].copy()

    return matrix


# --- CLEAN AUTOENCODER BUILD ---
def build_ae(input_matrix, latent_dim):
    input_dim = input_matrix.shape[1]
    
    # --- ENCODER ---
    # We remove the z_log_var and the Sampling layer entirely
    inputs = layers.Input(shape=(input_dim,))
    # Noise like VAE
    x = layers.GaussianNoise(0.1)(inputs)

    x = layers.Dense(2688, activation='relu')(x)
    x = layers.Dense(2176, activation='relu')(x)
    x = layers.Dense(1664, activation='relu')(x)
    x = layers.Dense(1152, activation='relu')(x)
    x = layers.Dense(768, activation='relu')(x)
    x = layers.Dense(384, activation='relu')(x)
    
    # Dropout to avoid overfitting
    x = layers.Dropout(0.2)(x)

    # The bottleneck is now just a single Dense layer
    latent_space = layers.Dense(latent_dim, name="latent_bottleneck")(x)
    
    encoder = keras.Model(inputs, latent_space, name="encoder")

    # --- DECODER ---
    decoder_input = layers.Input(shape=(latent_dim,))
    dx = layers.Dense(384, activation='relu')(decoder_input)
    dx = layers.Dense(768, activation='relu')(dx)
    dx = layers.Dense(1152, activation='relu')(dx)
    dx = layers.Dense(1664, activation='relu')(dx)
    dx = layers.Dense(2176, activation='relu')(dx)
    dx = layers.Dense(2688, activation='relu')(dx)
    decoder_outputs = layers.Dense(input_dim)(dx) 
    
    decoder = keras.Model(decoder_input, decoder_outputs, name="decoder")

    # --- FULL AUTOENCODER ---
    ae_outputs = decoder(encoder(inputs))
    autoencoder = keras.Model(inputs, ae_outputs, name="autoencoder")
    
    # SSE loss is equivalent to the VAE reconstruction loss
    # autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss='mse')
    autoencoder.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss=sum_squared_error
    )

    return autoencoder, encoder

# TF/Keras version skew: some builds expose register under utils, not saving.
try:
    _register_serializable = keras.saving.register_keras_serializable
except AttributeError:
    _register_serializable = keras.utils.register_keras_serializable


@_register_serializable()
def sum_squared_error(y_true, y_pred):
    # Per-sample sum over features; Keras then averages over the batch.
    # Identical to recon_loss in VaeLossLayer.
    return tf.reduce_sum(tf.square(y_true - y_pred), axis=-1)

# --- EXECUTION ---
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DATA = os.path.join(_REPO_ROOT, "Data", "ChemicalChecker_Full")
directory_path = _ensure_trailing_sep(
    os.environ.get("DRUGZIP_DATA_DIR", _DEFAULT_DATA)
)
print(f"Using data directory: {directory_path}")
input_matrix = build_input(directory_path)

_epochs = int(os.environ.get("DRUGZIP_EPOCHS", "500"))
_batch_size = int(os.environ.get("DRUGZIP_BATCH_SIZE", "256"))
_patience = int(os.environ.get("DRUGZIP_PATIENCE", "10"))
_run_name = os.environ.get("DRUGZIP_RUN_NAME", "ae-full-b256")

run = wandb.init(
    project="Paper-DrugZip",
    name=_run_name,
    config={
        "latent_dim": 128,
        "learning_rate": 1e-4,
        "epochs": _epochs,
        "batch_size": _batch_size,
        "layers": [2688, 2176, 1664, 1152, 768, 384],
        "scaling": None,
        "validation_fraction": 0.2,
        "split_seed": 42,
        "monitor": "val_loss",
        "patience": _patience,
        "start_from_epoch": 0,
        "max_samples": _max_samples_from_env(),
    }
)

# Same shuffle + split as the VAE script (same seed => identical train/val rows)
np.random.default_rng(run.config.split_seed).shuffle(input_matrix, axis=0)
split_at = int(len(input_matrix) * (1 - run.config.validation_fraction))
x_train, x_val = input_matrix[:split_at], input_matrix[split_at:]

latent_dimension = run.config.latent_dim
autoencoder, encoder = build_ae(input_matrix, latent_dimension)

early_stop = EarlyStopping(
    monitor=run.config.monitor,
    patience=run.config.patience,
    min_delta=0,
    restore_best_weights=True,
    start_from_epoch=run.config.start_from_epoch
)

print("\nStarting training for Standard Autoencoder...")
autoencoder.fit(
    x_train,
    x_train,
    validation_data=(x_val, x_val),   # AE needs (x, y); the VAE only needs (x,)
    shuffle=True,
    epochs=run.config.epochs,
    batch_size=run.config.batch_size,
    verbose=int(os.environ.get("DRUGZIP_VERBOSE", "0")),
    callbacks=[early_stop, WandbMetricsLogger()]
)

# Release the shuffled data and reload the original molecule order,
# otherwise the exported compressed vectors won't line up with the drugs.
del x_train, x_val, input_matrix
input_matrix = build_input(directory_path)

encoder.save("ae_encoder-do.keras")
autoencoder.save("ae_full.keras")
    
# Logs the model to W&B
run.save("ae_encoder-do.keras")

# Generate the compressed signatures
compressed_data = encoder.predict(input_matrix)

# Generate reconstructed data
reconstructed_data = autoencoder.predict(input_matrix)

# Save to CSV
df_25in1 = pd.DataFrame(compressed_data)
compressed_filename = f"ae-25in1-full-b{run.config.batch_size}.csv"
df_25in1.to_csv(compressed_filename, index=False)

# Save reconstructed CSV
df_reconstructed = pd.DataFrame(reconstructed_data)
reconstructed_filename = f"ae-25in1-reconstructed-full-{kl_weight}-b256.csv"
df_reconstructed.to_csv(reconstructed_filename, index=False)


# Save models in both formats
encoder_keras_filename = f"ae-encoder-full-{kl_weight}-b256.keras"
encoder_h5_filename = f"ae-encoder-full-{kl_weight}-b256.h5"
vae_keras_filename = f"ae-full-{kl_weight}-b256.keras"
vae_h5_filename = f"ae-full-{kl_weight}-b256.h5"

encoder.save(encoder_keras_filename)
encoder.save(encoder_h5_filename)
vae.save(vae_keras_filename)
vae.save(vae_h5_filename)

print("\nCSV generated with shape:", compressed_data.shape)
run.finish()