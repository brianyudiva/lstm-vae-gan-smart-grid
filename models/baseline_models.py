import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l1_l2
import numpy as np

@tf.keras.utils.register_keras_serializable()
class SamplingLayer(layers.Layer):    
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

def build_lstm_autoencoder(input_shape, latent_dim=32, lstm_units_1=32, lstm_units_2=16, 
                          dense_units_1=32, dense_units_2=16, dropout_rate=0.3, 
                          l1_reg=5e-4, l2_reg=1e-3):
    
    # === ENCODER ===
    encoder_input = layers.Input(shape=input_shape, name='encoder_input')
    
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
    
    latent_code = layers.Dense(latent_dim, activation='linear', name='latent_code', 
                         kernel_regularizer=l1_l2(l1_reg, l2_reg))(x)
    
    encoder = models.Model(encoder_input, latent_code, name='lstm_encoder')
    
    # === DECODER ===
    decoder_input = layers.Input(shape=(latent_dim,), name='decoder_input')
    
    x_dec = layers.RepeatVector(input_shape[0])(decoder_input)
    
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
    
    decoder = models.Model(decoder_input, decoder_output, name='lstm_decoder')
    
    # === COMBINED AUTOENCODER ===
    autoencoder_output = decoder(encoder(encoder_input))
    autoencoder = models.Model(encoder_input, autoencoder_output, name='lstm_autoencoder')
    
    return encoder, decoder, autoencoder


def build_vae_gan(input_shape, latent_dim=32, dense_units=[32, 16], 
                  dropout_rate=0.3, l1_reg=5e-4, l2_reg=1e-3):
    
    flattened_dim = input_shape[0] * input_shape[1]
    
    # === ENCODER ===
    encoder_input = layers.Input(shape=input_shape, name='vae_encoder_input')
    x = layers.Flatten()(encoder_input)
    
    for units in dense_units:
        x = layers.Dense(units, activation='relu', kernel_regularizer=l1_l2(l1_reg, l2_reg))(x)
        x = layers.Dropout(dropout_rate)(x)
    
    z_mean = layers.Dense(latent_dim, name='z_mean', kernel_regularizer=l1_l2(l1_reg, l2_reg))(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var', kernel_regularizer=l1_l2(l1_reg, l2_reg))(x)
    z = SamplingLayer(name='sampling')([z_mean, z_log_var])
    
    encoder = models.Model(encoder_input, [z_mean, z_log_var, z], name='vae_encoder')
    
    # === DECODER ===
    decoder_input = layers.Input(shape=(latent_dim,), name='vae_decoder_input')
    x_dec = decoder_input
    
    for units in reversed(dense_units):
        x_dec = layers.Dense(units, activation='relu', kernel_regularizer=l1_l2(l1_reg, l2_reg))(x_dec)
        x_dec = layers.Dropout(dropout_rate)(x_dec)
    
    x_dec = layers.Dense(flattened_dim, activation='linear', kernel_regularizer=l1_l2(l1_reg, l2_reg))(x_dec)
    decoder_output = layers.Reshape(input_shape)(x_dec)
    
    decoder = models.Model(decoder_input, decoder_output, name='vae_decoder')
    
    # === DISCRIMINATOR ===
    discriminator_input = layers.Input(shape=input_shape, name='discriminator_input')
    x_disc = layers.Flatten()(discriminator_input)
    
    for units in dense_units:
        x_disc = layers.Dense(units, activation='relu', kernel_regularizer=l1_l2(l1_reg, l2_reg))(x_disc)
        x_disc = layers.Dropout(dropout_rate)(x_disc)
    
    discriminator_output = layers.Dense(1, activation='sigmoid', name='discriminator_output')(x_disc)
    discriminator = models.Model(discriminator_input, discriminator_output, name='discriminator')
    
    # === COMBINED ===
    vae_output = decoder(encoder(encoder_input)[2])
    vae = models.Model(encoder_input, vae_output, name='vae_gan')
    
    return encoder, decoder, discriminator, vae


def build_lstm_gan(input_shape, latent_dim=32, lstm_units_1=32, lstm_units_2=16,
                   dense_units_1=32, dense_units_2=16, dropout_rate=0.3,
                   l1_reg=5e-4, l2_reg=1e-3):
    
    # === ENCODER ===
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
    
    latent_code = layers.Dense(latent_dim, activation='tanh', name='latent_code',
                              kernel_regularizer=l1_l2(l1_reg, l2_reg))(x)
    
    encoder = models.Model(encoder_input, latent_code, name='lstm_gan_encoder')
    
    # === DECODER ===
    decoder_input = layers.Input(shape=(latent_dim,), name='lstm_gan_decoder_input')

    x_dec = layers.RepeatVector(input_shape[0])(decoder_input)
    
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
    
    # === DISCRIMINATOR ===
    discriminator_input = layers.Input(shape=input_shape, name='lstm_gan_discriminator_input')

    x_disc = layers.Flatten()(discriminator_input)

    x_disc = layers.Dense(dense_units_1, activation='relu', kernel_regularizer=l1_l2(l1_reg, l2_reg))(x_disc)
    x_disc = layers.Dropout(dropout_rate)(x_disc)
    x_disc = layers.Dense(dense_units_2, activation='relu', kernel_regularizer=l1_l2(l1_reg, l2_reg))(x_disc)
    x_disc = layers.Dropout(dropout_rate)(x_disc)

    discriminator_output = layers.Dense(1, activation='sigmoid', name='discriminator_output')(x_disc)
    discriminator = models.Model(discriminator_input, discriminator_output, name='lstm_gan_discriminator')
    
    generator_output = decoder(encoder(encoder_input))
    generator = models.Model(encoder_input, generator_output, name='lstm_gan_generator')
    
    return encoder, decoder, discriminator, generator


def get_baseline_model(model_type, input_shape, **kwargs):
    if model_type == 'lstm_autoencoder':
        return build_lstm_autoencoder(input_shape, **kwargs)
    elif model_type == 'vae_gan':
        return build_vae_gan(input_shape, **kwargs)
    elif model_type == 'lstm_gan':
        return build_lstm_gan(input_shape, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
