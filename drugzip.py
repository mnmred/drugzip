# Making this VAE for CCSynergy.
# It will intake the 25 signatures of each drug and compress them into a smaller latent space.

import os
from unicodedata import name

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
import wandb
from wandb.integration.keras import WandbMetricsLogger

# Uses WANDB_API_KEY from the environment when present
if os.environ.get("WANDB_API_KEY"):
    wandb.login(key=os.environ["WANDB_API_KEY"])


def _ensure_trailing_sep(path: str) -> str:
    return path if path.endswith(("/", "\\")) else path + os.sep


def _out_dir() -> str:
    path = os.environ.get("DRUGZIP_OUT_DIR", "results")
    os.makedirs(path, exist_ok=True)
    return path


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


def build_vae(input_matrix, latent_dim, kl_weight=1.0):
    input_dim = input_matrix.shape[1]
    
    # --- ENCODER ---
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(2688, activation='relu')(inputs) #21x128
    x = layers.Dense(2176, activation='relu')(x) #17x128
    x = layers.Dense(1664, activation='relu')(x) #13x128
    x = layers.Dense(1152, activation='relu')(x) #9x128
    x = layers.Dense(768, activation='relu')(x) #6x128
    x = layers.Dense(384, activation='relu')(x) #3x128
    
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

    # --- SAMPLING LAYER ---
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    z = layers.Lambda(sampling, name="z")([z_mean, z_log_var])

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
    outputs = decoder(z)

    # --- VAE LOSS LAYER (FOR KERAS 3) ---
    class VaeLossLayer(layers.Layer):
        def __init__(self, kl_weight=1.0, **kwargs):
            super().__init__(**kwargs)
            self.kl_weight = kl_weight

            # Create trackers
            self.recon_tracker = keras.metrics.Mean(name="recon_loss")
            self.kl_tracker = keras.metrics.Mean(name="kl_loss")
            self.total_tracker = keras.metrics.Mean(name="total_loss")

            # Sigma trackers
            self.sigma_mean_tracker = keras.metrics.Mean(name="sigma_mean")
            self.sigma_std_tracker = keras.metrics.Mean(name="sigma_std")
            self.sigma_min_tracker = keras.metrics.Mean(name="sigma_min")
            self.sigma_max_tracker = keras.metrics.Mean(name="sigma_max")

        @property
        def metrics(self):
            return [
                self.recon_tracker,
                self.kl_tracker,
                self.total_tracker,
                self.sigma_mean_tracker,
                self.sigma_std_tracker,
                self.sigma_min_tracker,
                self.sigma_max_tracker,
        ]

        def call(self, inputs_list):
            x_in, x_out, z_m, z_lv = inputs_list
            
            # Reconstruction Loss
            recon_loss = tf.reduce_mean(tf.reduce_sum(tf.square(x_in - x_out), axis=-1))
            
            # KL Loss
            kl = -0.5 * tf.reduce_sum(1 + z_lv - tf.square(z_m) - tf.exp(z_lv), axis=-1)
            kl = tf.reduce_mean(kl)

            # Sigma
            sigma = tf.exp(0.5 * z_lv)

            sigma_mean = tf.reduce_mean(sigma)
            sigma_std = tf.math.reduce_std(sigma)
            sigma_min = tf.reduce_min(sigma)
            sigma_max = tf.reduce_max(sigma)
            
            # Total Loss
            total_loss = recon_loss + self.kl_weight * kl
            
            # 2. Add loss and update trackers
            self.add_loss(total_loss)
            self.recon_tracker.update_state(recon_loss)
            self.kl_tracker.update_state(kl)
            self.total_tracker.update_state(total_loss)

            self.sigma_mean_tracker.update_state(sigma_mean)
            self.sigma_std_tracker.update_state(sigma_std)
            self.sigma_min_tracker.update_state(sigma_min)
            self.sigma_max_tracker.update_state(sigma_max)


            return x_out

        # 3. Required for Keras 3+ to understand output shapes
        def compute_output_shape(self, input_shape):
            # The output shape is the same as the second input (x_out)
            return input_shape[1]

    outputs_with_loss = VaeLossLayer(kl_weight=kl_weight)([inputs, outputs, z_mean, z_log_var])
    
    vae = keras.Model(inputs, outputs_with_loss, name="vae")
    
    # Compile with the metrics included
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4))

    # Encoder for extraction
    encoder = keras.Model(inputs, z_mean, name="encoder")

    return vae, encoder

# --- EXECUTION VAE ---
kl_weights = [0.0]  # Different beta values to test
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
_verbose = int(os.environ.get("DRUGZIP_VERBOSE", "0"))

for kl_weight in kl_weights:
    # INITIALIZE W&B RUN
    _run_name = os.environ.get(
        "DRUGZIP_RUN_NAME", f"vae-full-kl-{kl_weight}-b{_batch_size}"
    )
    run = wandb.init(
        project="Paper-DrugZip",
        name=_run_name,
        config={
            "kl_weight": kl_weight,
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

    # Shuffle rows once, then split with views to avoid copying the matrix.
    np.random.default_rng(run.config.split_seed).shuffle(input_matrix, axis=0)
    split_at = int(len(input_matrix) * (1 - run.config.validation_fraction))
    x_train, x_val = input_matrix[:split_at], input_matrix[split_at:]

    latent_dimension = run.config.latent_dim # Compression target

    vae, encoder = build_vae(input_matrix, latent_dimension, kl_weight)

    early_stop = EarlyStopping(
        monitor=run.config.monitor,
        patience=run.config.patience,
        min_delta=0,
        restore_best_weights=True,
        start_from_epoch=run.config.start_from_epoch
    )

    # Training: Use a batch size accordingly to the dataset size. For 1.2M samples, a batch size of 256 is reasonable.
    print(f"\nStarting training for Target Beta = {kl_weight}... ")
    history = vae.fit(
        x_train,
        validation_data=(x_val,),
        shuffle=True,
        epochs=run.config.epochs,
        batch_size=run.config.batch_size,
        verbose=_verbose,
        callbacks=[early_stop, WandbMetricsLogger()]
    )

    # Release the shuffled training data and reload the original molecule order.
    del x_train, x_val, input_matrix
    input_matrix = build_input(directory_path)

    out = _out_dir()

    # Generate the compressed signatures
    compressed_data = encoder.predict(input_matrix)

    # Generate reconstructed data (decoder output)
    reconstructed_data = vae.predict(input_matrix)

    # Save to CSV
    df_25in1 = pd.DataFrame(compressed_data)
    compressed_filename = os.path.join(
        out, f"drugzip-25in1-full-{kl_weight}-b{run.config.batch_size}.csv"
    )
    df_25in1.to_csv(compressed_filename, index=False)

    # Save reconstructed CSV
    df_reconstructed = pd.DataFrame(reconstructed_data)
    reconstructed_filename = os.path.join(
        out, f"drugzip-reconstructed-full-{kl_weight}-b256.csv"
    )
    df_reconstructed.to_csv(reconstructed_filename, index=False)

    # Save models in both formats
    encoder_keras_filename = os.path.join(out, f"encoder-full-{kl_weight}-b256.keras")
    encoder_h5_filename = os.path.join(out, f"encoder-full-{kl_weight}-b256.h5")
    vae_keras_filename = os.path.join(out, f"vae-full-{kl_weight}-b256.keras")
    vae_h5_filename = os.path.join(out, f"vae-full-{kl_weight}-b256.h5")

    encoder.save(encoder_keras_filename)
    encoder.save(encoder_h5_filename)
    vae.save(vae_keras_filename)
    vae.save(vae_h5_filename)

    # Prefer run.save over log_artifact: artifact staging does chmod, which
    # fails on TU Delft NFS (PermissionError / Operation not permitted).
    for _path in (
        compressed_filename,
        reconstructed_filename,
        encoder_keras_filename,
        encoder_h5_filename,
        vae_keras_filename,
        vae_h5_filename,
    ):
        run.save(_path, policy="now")

    print("\nCSV generated with shape:", compressed_data.shape)
    print(f"Reconstructed data saved with shape: {reconstructed_data.shape}")
    print(f"Outputs written under {out}/")
    print(f"All files saved to wandb run")
    run.finish()
    
    del compressed_data, reconstructed_data, df_25in1, df_reconstructed