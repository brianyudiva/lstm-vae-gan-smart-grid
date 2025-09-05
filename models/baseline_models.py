import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l1_l2
import numpy as np

# === CUSTOM SAMPLING LAYER FOR VAE SERIALIZATION ===
class SamplingLayer(layers.Layer):
    """Custom sampling layer for VAE that supports serialization"""
    
    def __init__(self, name='sampling', **kwargs):
        super(SamplingLayer, self).__init__(name=name, **kwargs)
    
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    def get_config(self):
        config = super(SamplingLayer, self).get_config()
        return config

# === BASELINE MODELS FOR COMPARISON ===

def build_lstm_autoencoder(input_shape, latent_dim=32, lstm_units_1=32, lstm_units_2=16, 
                          dense_units_1=32, dense_units_2=16, dropout_rate=0.3, 
                          l1_reg=5e-4, l2_reg=1e-3):
    """
    Pure LSTM Autoencoder (No VAE, No GAN)
    Tests: "Do we need probabilistic modeling at all?"
    """
    
    # === ENCODER ===
    encoder_input = layers.Input(shape=input_shape, name='encoder_input')
    
    # LSTM layers
    x = layers.LSTM(
        lstm_units_1,
        return_sequences=True,
        kernel_regularizer=l1_l2(l1_reg, l2_reg),
        recurrent_regularizer=l1_l2(l1_reg, l2_reg),
        dropout=dropout_rate,
        recurrent_dropout=dropout_rate
    )(encoder_input)
    
    x = layers.LSTM(
        lstm_units_2,
        return_sequences=False,
        kernel_regularizer=l1_l2(l1_reg, l2_reg),
        recurrent_regularizer=l1_l2(l1_reg, l2_reg),
        dropout=dropout_rate,
        recurrent_dropout=dropout_rate
    )(x)
    
    # Dense layers
    x = layers.Dense(dense_units_1, activation='relu', kernel_regularizer=l1_l2(l1_reg, l2_reg))(x)
    x = layers.Dropout(dropout_rate * 0.67)(x)
    x = layers.Dense(dense_units_2, activation='relu', kernel_regularizer=l1_l2(l1_reg, l2_reg))(x)
    x = layers.Dropout(dropout_rate * 0.67)(x)
    
    # Latent representation (deterministic)
    latent = layers.Dense(latent_dim, activation='linear', name='latent_code', 
                         kernel_regularizer=l1_l2(l1_reg, l2_reg))(x)
    
    encoder = models.Model(encoder_input, latent, name='lstm_encoder')
    
    # === DECODER ===
    decoder_input = layers.Input(shape=(latent_dim,), name='decoder_input')
    
    # Expand latent to dense
    x_dec = layers.Dense(dense_units_2, activation='relu', kernel_regularizer=l1_l2(l1_reg, l2_reg))(decoder_input)
    x_dec = layers.Dropout(dropout_rate * 0.67)(x_dec)
    x_dec = layers.Dense(dense_units_1, activation='relu', kernel_regularizer=l1_l2(l1_reg, l2_reg))(x_dec)
    x_dec = layers.Dropout(dropout_rate * 0.67)(x_dec)
    
    # Prepare for LSTM
    x_dec = layers.Dense(lstm_units_2, activation='relu', kernel_regularizer=l1_l2(l1_reg, l2_reg))(x_dec)
    x_dec = layers.RepeatVector(input_shape[0])(x_dec)
    
    # LSTM layers
    x_dec = layers.LSTM(
        lstm_units_2,
        return_sequences=True,
        kernel_regularizer=l1_l2(l1_reg, l2_reg),
        recurrent_regularizer=l1_l2(l1_reg, l2_reg),
        dropout=dropout_rate,
        recurrent_dropout=dropout_rate
    )(x_dec)
    
    x_dec = layers.LSTM(
        lstm_units_1,
        return_sequences=True,
        kernel_regularizer=l1_l2(l1_reg, l2_reg),
        recurrent_regularizer=l1_l2(l1_reg, l2_reg),
        dropout=dropout_rate,
        recurrent_dropout=dropout_rate
    )(x_dec)
    
    # Output layer
    decoder_output = layers.TimeDistributed(
        layers.Dense(input_shape[1], activation='linear', kernel_regularizer=l1_l2(l1_reg, l2_reg))
    )(x_dec)
    
    decoder = models.Model(decoder_input, decoder_output, name='lstm_decoder')
    
    # === COMBINED AUTOENCODER ===
    autoencoder_output = decoder(encoder(encoder_input))
    autoencoder = models.Model(encoder_input, autoencoder_output, name='lstm_autoencoder')
    
    return encoder, decoder, autoencoder


def build_vae_gan(input_shape, latent_dim=32, dense_units=[128, 64, 32], 
                  dropout_rate=0.3, l1_reg=5e-4, l2_reg=1e-3):
    """
    VAE-GAN with Dense Layers (No LSTM)
    Tests: "Do we need temporal sequence modeling?"
    """
    
    # Flatten input for dense layers
    flattened_dim = input_shape[0] * input_shape[1]
    
    # === VAE ENCODER ===
    encoder_input = layers.Input(shape=input_shape, name='vae_encoder_input')
    x = layers.Flatten()(encoder_input)
    
    for units in dense_units:
        x = layers.Dense(units, activation='relu', kernel_regularizer=l1_l2(l1_reg, l2_reg))(x)
        x = layers.Dropout(dropout_rate)(x)
    
    # VAE latent parameters
    z_mean = layers.Dense(latent_dim, name='z_mean', kernel_regularizer=l1_l2(l1_reg, l2_reg))(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var', kernel_regularizer=l1_l2(l1_reg, l2_reg))(x)
    
    # Sampling layer
    z = SamplingLayer(name='sampling')([z_mean, z_log_var])
    
    encoder = models.Model(encoder_input, [z_mean, z_log_var, z], name='vae_encoder')
    
    # === VAE DECODER ===
    decoder_input = layers.Input(shape=(latent_dim,), name='vae_decoder_input')
    x_dec = decoder_input
    
    for units in reversed(dense_units):
        x_dec = layers.Dense(units, activation='relu', kernel_regularizer=l1_l2(l1_reg, l2_reg))(x_dec)
        x_dec = layers.Dropout(dropout_rate)(x_dec)
    
    x_dec = layers.Dense(flattened_dim, activation='linear', kernel_regularizer=l1_l2(l1_reg, l2_reg))(x_dec)
    decoder_output = layers.Reshape(input_shape)(x_dec)
    
    decoder = models.Model(decoder_input, decoder_output, name='vae_decoder')
    
    # === GAN DISCRIMINATOR ===
    discriminator_input = layers.Input(shape=input_shape, name='discriminator_input')
    x_disc = layers.Flatten()(discriminator_input)
    
    for units in dense_units:
        x_disc = layers.Dense(units, activation='relu', kernel_regularizer=l1_l2(l1_reg, l2_reg))(x_disc)
        x_disc = layers.Dropout(dropout_rate)(x_disc)
    
    discriminator_output = layers.Dense(1, activation='sigmoid', name='discriminator_output')(x_disc)
    discriminator = models.Model(discriminator_input, discriminator_output, name='discriminator')
    
    # === COMBINED VAE ===
    vae_output = decoder(encoder(encoder_input)[2])  # Use sampled z
    vae = models.Model(encoder_input, vae_output, name='vae_gan')
    
    return encoder, decoder, discriminator, vae


def build_lstm_gan(input_shape, latent_dim=32, lstm_units_1=32, lstm_units_2=16,
                   dense_units_1=32, dense_units_2=16, dropout_rate=0.3,
                   l1_reg=5e-4, l2_reg=1e-3):
    """
    LSTM-GAN (No VAE)
    Tests: "Do we need probabilistic VAE, or is deterministic GAN enough?"
    """
    
    # === LSTM ENCODER ===
    encoder_input = layers.Input(shape=input_shape, name='lstm_gan_encoder_input')
    
    x = layers.LSTM(
        lstm_units_1,
        return_sequences=True,
        kernel_regularizer=l1_l2(l1_reg, l2_reg),
        recurrent_regularizer=l1_l2(l1_reg, l2_reg),
        dropout=dropout_rate,
        recurrent_dropout=dropout_rate
    )(encoder_input)
    
    x = layers.LSTM(
        lstm_units_2,
        return_sequences=False,
        kernel_regularizer=l1_l2(l1_reg, l2_reg),
        recurrent_regularizer=l1_l2(l1_reg, l2_reg),
        dropout=dropout_rate,
        recurrent_dropout=dropout_rate
    )(x)
    
    x = layers.Dense(dense_units_1, activation='relu', kernel_regularizer=l1_l2(l1_reg, l2_reg))(x)
    x = layers.Dropout(dropout_rate * 0.67)(x)
    x = layers.Dense(dense_units_2, activation='relu', kernel_regularizer=l1_l2(l1_reg, l2_reg))(x)
    x = layers.Dropout(dropout_rate * 0.67)(x)
    
    # Deterministic latent code (no sampling)
    latent_code = layers.Dense(latent_dim, activation='tanh', name='latent_code',
                              kernel_regularizer=l1_l2(l1_reg, l2_reg))(x)
    
    encoder = models.Model(encoder_input, latent_code, name='lstm_gan_encoder')
    
    # === LSTM DECODER (GENERATOR) ===
    decoder_input = layers.Input(shape=(latent_dim,), name='lstm_gan_decoder_input')
    
    x_dec = layers.Dense(dense_units_2, activation='relu', kernel_regularizer=l1_l2(l1_reg, l2_reg))(decoder_input)
    x_dec = layers.Dropout(dropout_rate * 0.67)(x_dec)
    x_dec = layers.Dense(dense_units_1, activation='relu', kernel_regularizer=l1_l2(l1_reg, l2_reg))(x_dec)
    x_dec = layers.Dropout(dropout_rate * 0.67)(x_dec)
    
    x_dec = layers.Dense(lstm_units_2, activation='relu', kernel_regularizer=l1_l2(l1_reg, l2_reg))(x_dec)
    x_dec = layers.RepeatVector(input_shape[0])(x_dec)
    
    x_dec = layers.LSTM(
        lstm_units_2,
        return_sequences=True,
        kernel_regularizer=l1_l2(l1_reg, l2_reg),
        recurrent_regularizer=l1_l2(l1_reg, l2_reg),
        dropout=dropout_rate,
        recurrent_dropout=dropout_rate
    )(x_dec)
    
    x_dec = layers.LSTM(
        lstm_units_1,
        return_sequences=True,
        kernel_regularizer=l1_l2(l1_reg, l2_reg),
        recurrent_regularizer=l1_l2(l1_reg, l2_reg),
        dropout=dropout_rate,
        recurrent_dropout=dropout_rate
    )(x_dec)
    
    decoder_output = layers.TimeDistributed(
        layers.Dense(input_shape[1], activation='linear', kernel_regularizer=l1_l2(l1_reg, l2_reg))
    )(x_dec)
    
    decoder = models.Model(decoder_input, decoder_output, name='lstm_gan_decoder')
    
    # === GAN DISCRIMINATOR ===
    discriminator_input = layers.Input(shape=input_shape, name='lstm_gan_discriminator_input')
    
    # Use LSTM for discriminator too
    x_disc = layers.LSTM(
        lstm_units_1,
        return_sequences=True,
        kernel_regularizer=l1_l2(l1_reg, l2_reg),
        dropout=dropout_rate
    )(discriminator_input)
    
    x_disc = layers.LSTM(
        lstm_units_2,
        return_sequences=False,
        kernel_regularizer=l1_l2(l1_reg, l2_reg),
        dropout=dropout_rate
    )(x_disc)
    
    x_disc = layers.Dense(dense_units_1, activation='relu', kernel_regularizer=l1_l2(l1_reg, l2_reg))(x_disc)
    x_disc = layers.Dropout(dropout_rate)(x_disc)
    x_disc = layers.Dense(dense_units_2, activation='relu', kernel_regularizer=l1_l2(l1_reg, l2_reg))(x_disc)
    x_disc = layers.Dropout(dropout_rate)(x_disc)
    
    discriminator_output = layers.Dense(1, activation='sigmoid', name='discriminator_output')(x_disc)
    discriminator = models.Model(discriminator_input, discriminator_output, name='lstm_gan_discriminator')
    
    # === COMBINED GENERATOR ===
    generator_output = decoder(encoder(encoder_input))
    generator = models.Model(encoder_input, generator_output, name='lstm_gan_generator')
    
    return encoder, decoder, discriminator, generator


def build_lstm_vae(input_shape, latent_dim=32, lstm_units_1=32, lstm_units_2=16,
                   dense_units_1=32, dense_units_2=16, dropout_rate=0.3,
                   l1_reg=5e-4, l2_reg=1e-3):
    """
    LSTM-VAE (No GAN)
    Tests: "Do we need adversarial training, or is VAE sufficient?"
    """
    
    # === VAE ENCODER ===
    encoder_input = layers.Input(shape=input_shape, name='lstm_vae_encoder_input')
    
    x = layers.LSTM(
        lstm_units_1,
        return_sequences=True,
        kernel_regularizer=l1_l2(l1_reg, l2_reg),
        recurrent_regularizer=l1_l2(l1_reg, l2_reg),
        dropout=dropout_rate,
        recurrent_dropout=dropout_rate
    )(encoder_input)
    
    x = layers.LSTM(
        lstm_units_2,
        return_sequences=False,
        kernel_regularizer=l1_l2(l1_reg, l2_reg),
        recurrent_regularizer=l1_l2(l1_reg, l2_reg),
        dropout=dropout_rate,
        recurrent_dropout=dropout_rate
    )(x)
    
    x = layers.Dense(dense_units_1, activation='relu', kernel_regularizer=l1_l2(l1_reg, l2_reg))(x)
    x = layers.Dropout(dropout_rate * 0.67)(x)
    x = layers.Dense(dense_units_2, activation='relu', kernel_regularizer=l1_l2(l1_reg, l2_reg))(x)
    x = layers.Dropout(dropout_rate * 0.67)(x)
    
    # VAE latent parameters
    z_mean = layers.Dense(latent_dim, name='z_mean', kernel_regularizer=l1_l2(l1_reg, l2_reg))(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var', kernel_regularizer=l1_l2(l1_reg, l2_reg))(x)
    z = SamplingLayer(name='sampling')([z_mean, z_log_var])
    
    encoder = models.Model(encoder_input, [z_mean, z_log_var, z], name='lstm_vae_encoder')
    
    # === VAE DECODER ===
    decoder_input = layers.Input(shape=(latent_dim,), name='lstm_vae_decoder_input')
    
    x_dec = layers.Dense(dense_units_2, activation='relu', kernel_regularizer=l1_l2(l1_reg, l2_reg))(decoder_input)
    x_dec = layers.Dropout(dropout_rate * 0.67)(x_dec)
    x_dec = layers.Dense(dense_units_1, activation='relu', kernel_regularizer=l1_l2(l1_reg, l2_reg))(x_dec)
    x_dec = layers.Dropout(dropout_rate * 0.67)(x_dec)
    
    x_dec = layers.Dense(lstm_units_2, activation='relu', kernel_regularizer=l1_l2(l1_reg, l2_reg))(x_dec)
    x_dec = layers.RepeatVector(input_shape[0])(x_dec)
    
    x_dec = layers.LSTM(
        lstm_units_2,
        return_sequences=True,
        kernel_regularizer=l1_l2(l1_reg, l2_reg),
        recurrent_regularizer=l1_l2(l1_reg, l2_reg),
        dropout=dropout_rate,
        recurrent_dropout=dropout_rate
    )(x_dec)
    
    x_dec = layers.LSTM(
        lstm_units_1,
        return_sequences=True,
        kernel_regularizer=l1_l2(l1_reg, l2_reg),
        recurrent_regularizer=l1_l2(l1_reg, l2_reg),
        dropout=dropout_rate,
        recurrent_dropout=dropout_rate
    )(x_dec)
    
    decoder_output = layers.TimeDistributed(
        layers.Dense(input_shape[1], activation='linear', kernel_regularizer=l1_l2(l1_reg, l2_reg))
    )(x_dec)
    
    decoder = models.Model(decoder_input, decoder_output, name='lstm_vae_decoder')
    
    # === COMBINED VAE ===
    vae_output = decoder(encoder(encoder_input)[2])  # Use sampled z
    vae = models.Model(encoder_input, vae_output, name='lstm_vae')
    
    return encoder, decoder, vae


# === MODEL FACTORY ===
def get_baseline_model(model_type, input_shape, **kwargs):
    """
    Factory function to get any baseline model
    
    Args:
        model_type: 'lstm_autoencoder', 'vae_gan', 'lstm_gan', 'lstm_vae'
        input_shape: (timesteps, features)
        **kwargs: Model-specific parameters
    
    Returns:
        Tuple of model components based on type
    """
    
    if model_type == 'lstm_autoencoder':
        return build_lstm_autoencoder(input_shape, **kwargs)
    elif model_type == 'vae_gan':
        return build_vae_gan(input_shape, **kwargs)
    elif model_type == 'lstm_gan':
        return build_lstm_gan(input_shape, **kwargs)
    elif model_type == 'lstm_vae':
        return build_lstm_vae(input_shape, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# === MODEL SUMMARY FUNCTION ===
def print_model_comparison(input_shape=(24, 26)):
    """Print parameter comparison of all models"""
    
    print("=" * 80)
    print("BASELINE MODELS PARAMETER COMPARISON")
    print("=" * 80)
    
    latent_dim = 32
    models_info = []
    
    # LSTM Autoencoder
    encoder, decoder, autoencoder = build_lstm_autoencoder(input_shape, latent_dim)
    total_params = autoencoder.count_params()
    models_info.append(('LSTM Autoencoder', total_params, 'Reconstruction loss only'))
    
    # VAE-GAN
    encoder, decoder, discriminator, vae = build_vae_gan(input_shape, latent_dim)
    total_params = vae.count_params() + discriminator.count_params()
    models_info.append(('VAE-GAN', total_params, 'Dense layers + VAE + GAN'))
    
    # LSTM-GAN
    encoder, decoder, discriminator, generator = build_lstm_gan(input_shape, latent_dim)
    total_params = generator.count_params() + discriminator.count_params()
    models_info.append(('LSTM-GAN', total_params, 'LSTM + deterministic + GAN'))
    
    # LSTM-VAE
    encoder, decoder, vae = build_lstm_vae(input_shape, latent_dim)
    total_params = vae.count_params()
    models_info.append(('LSTM-VAE', total_params, 'LSTM + VAE (no GAN)'))
    
    # Print comparison
    for name, params, description in models_info:
        print(f"{name:<20} | {params:>8,} params | {description}")
    
    print("=" * 80)


if __name__ == "__main__":
    # Test all models
    print_model_comparison()
