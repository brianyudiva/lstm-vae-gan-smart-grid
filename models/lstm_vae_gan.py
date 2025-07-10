import tensorflow as tf
from tensorflow.python.keras import layers, models

# === LSTM Encoder ===
def build_encoder(input_shape, latent_dim):
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.LSTM(64, return_sequences=True)(inputs)
    x = layers.LSTM(32)(x)
    latent = layers.Dense(latent_dim)(x)
    return tf.keras.Model(inputs, latent, name="Encoder")

# === LSTM Decoder ===
def build_decoder(sequence_length, feature_dim, latent_dim):
    inputs = tf.keras.Input(shape=(latent_dim,))
    x = layers.RepeatVector(sequence_length)(inputs)
    x = layers.LSTM(32, return_sequences=True)(x)
    x = layers.LSTM(64, return_sequences=True)(x)
    outputs = layers.TimeDistributed(layers.Dense(feature_dim))(x)
    return tf.keras.Model(inputs, outputs, name="Decoder")

# === Discriminator ===
def build_discriminator(sequence_length, feature_dim):
    inputs = tf.keras.Input(shape=(sequence_length, feature_dim))
    x = layers.LSTM(64, return_sequences=True)(inputs)
    x = layers.LSTM(32)(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    return tf.keras.Model(inputs, x, name="Discriminator")

# === Combined GAN ===
def build_lstm_vae_gan(input_shape, latent_dim):
    encoder = build_encoder(input_shape, latent_dim)
    decoder = build_decoder(input_shape[0], input_shape[1], latent_dim)
    discriminator = build_discriminator(input_shape[0], input_shape[1])
    return encoder, decoder, discriminator
