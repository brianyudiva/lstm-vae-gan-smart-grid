import numpy as np
import tensorflow as tf
from models.lstm_vae_gan import build_lstm_vae_gan
import os

# === CONFIG ===
latent_dim = 16
sequence_path = "data/sequences"
output_path = "outputs/checkpoints"
os.makedirs(output_path, exist_ok=True)

# === LOAD DATA ===
X_train = np.load(f"{sequence_path}/X_train.npy")
X_test = np.load(f"{sequence_path}/X_test.npy")

# === BUILD MODELS ===
input_shape = (X_train.shape[1], X_train.shape[2])
encoder, decoder, discriminator = build_lstm_vae_gan(input_shape, latent_dim)

generator_input = tf.keras.Input(shape=input_shape)
latent = encoder(generator_input)
reconstructed = decoder(latent)

discriminator.trainable = False
discriminator_output = discriminator(reconstructed)

van_gan = tf.keras.Model(generator_input, discriminator_output, name="LSTM-VAN-GAN")

# === COMPILE MODELS ===
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

bce = tf.keras.losses.BinaryCrossentropy()

# === TRAIN LOOP ===
batch_size = 32
epochs = 30
steps_per_epoch = X_train.shape[0] // batch_size

for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    for step in range(steps_per_epoch):
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_seq = X_train[idx]

        # === Train Discriminator ===
        latent = encoder(real_seq)
        fake_seq = decoder(latent)

        real_label = tf.ones((batch_size, 1))
        fake_label = tf.zeros((batch_size, 1))

        with tf.GradientTape() as tape:
            real_pred = discriminator(real_seq)
            fake_pred = discriminator(fake_seq)
            d_loss_real = bce(real_label, real_pred)
            d_loss_fake = bce(fake_label, fake_pred)
            d_loss = d_loss_real + d_loss_fake

        grads = tape.gradient(d_loss, discriminator.trainable_weights)
        discriminator_optimizer.apply_gradients(zip(grads, discriminator.trainable_weights))

        # === Train Generator ===
        misleading_label = tf.ones((batch_size, 1))

        with tf.GradientTape() as tape:
            latent = encoder(real_seq)
            reconstructed = decoder(latent)
            output = discriminator(reconstructed)
            g_loss = bce(misleading_label, output)

        trainable_weights = encoder.trainable_weights + decoder.trainable_weights
        grads = tape.gradient(g_loss, trainable_weights)
        generator_optimizer.apply_gradients(zip(grads, trainable_weights))

        if step % 100 == 0:
            print(f"Step {step}: D_loss = {d_loss.numpy():.4f}, G_loss = {g_loss.numpy():.4f}")

# === SAVE MODELS ===
encoder.save(f"{output_path}/encoder.h5")
decoder.save(f"{output_path}/decoder.h5")
discriminator.save(f"{output_path}/discriminator.h5")
