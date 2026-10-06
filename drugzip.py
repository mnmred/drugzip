# Making this VAE for CCSynergy.
# It will intake the 25 signatures of each drug and compress them into a smaller latent space.

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

wandb.login(key="INSERT_YOUR_KEY_HERE")


# Usually in Matrix MxN, M is number of samples, N is number of features. This adapts our 
# signature matrix accordingly.
def build_input(directory_path):
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
            # 1. Create trackers
            self.recon_tracker = keras.metrics.Mean(name="recon_loss")
            self.kl_tracker = keras.metrics.Mean(name="kl_loss")
            self.total_tracker = keras.metrics.Mean(name="total_loss")

        def call(self, inputs_list):
            x_in, x_out, z_m, z_lv = inputs_list
            
            # Reconstruction Loss
            recon_loss = tf.reduce_mean(tf.reduce_sum(tf.square(x_in - x_out), axis=-1))
            
            # KL Loss
            kl = -0.5 * tf.reduce_sum(1 + z_lv - tf.square(z_m) - tf.exp(z_lv), axis=-1)
            kl = tf.reduce_mean(kl)
            
            # Total Loss
            total_loss = recon_loss + self.kl_weight * kl
            
            # 2. Add loss and update trackers
            self.add_loss(total_loss)
            self.recon_tracker.update_state(recon_loss)
            self.kl_tracker.update_state(kl)
            self.total_tracker.update_state(total_loss)

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
kl_weights = [0.0, 0.5, 1.0, 1.5, 2.0]  # Different beta values to test
directory_path = "Data/Drug_Representation/Sanger_Full/" # Replace with your actual path to the CSV files
input_matrix = build_input(directory_path)
for kl_weight in kl_weights:
    # INITIALIZE W&B RUN
    run = wandb.init(
        project="CCSynergy-VAE",
        name=f"vae-full-kl-{kl_weight}",
        config={
            "kl_weight": kl_weight,
            "latent_dim": 128,
            "learning_rate": 1e-4,
            "epochs": 500,
            "batch_size": 8,
            "layers": [2688, 2176, 1664, 1152, 768, 384]
        }
    )

    latent_dimension = run.config.latent_dim # Compression target

    vae, encoder = build_vae(input_matrix, latent_dimension, kl_weight)

    early_stop = EarlyStopping(
        monitor='loss', 
        patience=10, 
        restore_best_weights=True
    )

    # Training: Use a small batch size since we only have 62 drugs
    print("Starting training... Keep an eye on the 'loss' value below.")
    history = vae.fit(
        input_matrix, 
        input_matrix, 
        epochs=500,          # Set high; EarlyStopping will act as the "brake"
        batch_size=8, 
        verbose=0,           # 1 shows the progress bar and the Loss per epoch
        callbacks=[early_stop, WandbMetricsLogger()]
    )

    # Generate the compressed signatures
    compressed_data = encoder.predict(input_matrix)

    # Save to CSV
    df_25in1 = pd.DataFrame(compressed_data)
    filename = f"drugzip-25in1-full-{kl_weight}.csv"
    df_25in1.to_csv(filename, index=False)

    print("\nCSV generated with shape:", compressed_data.shape)
    # plot_drug_signatures(compressed_data, kl_weight)
    run.finish()