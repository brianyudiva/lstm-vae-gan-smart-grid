import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l1_l2
import numpy as np

def sampling_layer(z_mean, z_log_var, name='sampling'):
    def sampling(args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    latent_dim = z_mean.shape[-1]
    return layers.Lambda(sampling, output_shape=(latent_dim,), name=name)([z_mean, z_log_var])

def build_lstm_vae_gan_regular(input_shape, latent_dim=8):
    l1_reg = 5e-4
    l2_reg = 1e-3
    dropout_rate = 0.3
    
    # === ENCODER ===
    encoder_input = layers.Input(shape=input_shape, name='input_layer')
    
    x = layers.LSTM(
        16,
        return_sequences=True, 
        kernel_regularizer=l1_l2(l1_reg, l2_reg),
        recurrent_regularizer=l1_l2(l1_reg, l2_reg),
        dropout=dropout_rate,
        recurrent_dropout=dropout_rate
    )(encoder_input)
    
    x = layers.LSTM(
        8, 
        return_sequences=False, 
        kernel_regularizer=l1_l2(l1_reg, l2_reg),
        recurrent_regularizer=l1_l2(l1_reg, l2_reg),
        dropout=dropout_rate,
        recurrent_dropout=dropout_rate
    )(x)
    
    z_mean = layers.Dense(latent_dim, name='z_mean', kernel_regularizer=l1_l2(l1_reg, l2_reg))(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var', kernel_regularizer=l1_l2(l1_reg, l2_reg))(x)
    z = sampling_layer(z_mean, z_log_var, 'sampling')
    
    encoder = models.Model(encoder_input, [z_mean, z_log_var, z], name='regular_encoder')
    
    # === DECODER ===
    decoder_input = layers.Input(shape=(latent_dim,))
    
    x = layers.RepeatVector(input_shape[0])(decoder_input)
    
    x = layers.LSTM(
        8, 
        return_sequences=True,
        kernel_regularizer=l1_l2(l1_reg, l2_reg),
        recurrent_regularizer=l1_l2(l1_reg, l2_reg),
        dropout=dropout_rate,
        recurrent_dropout=dropout_rate
    )(x)
    
    x = layers.LSTM(
        16, 
        return_sequences=True,
        kernel_regularizer=l1_l2(l1_reg, l2_reg),
        recurrent_regularizer=l1_l2(l1_reg, l2_reg),
        dropout=dropout_rate,
        recurrent_dropout=dropout_rate
    )(x)
    
    decoder_output = layers.Dense(input_shape[1], kernel_regularizer=l1_l2(l1_reg, l2_reg))(x)
    decoder = models.Model(decoder_input, decoder_output, name='regular_decoder')
    
    # === DISCRIMINATOR ===
    discriminator_input = layers.Input(shape=input_shape)
    
    x = layers.Dense(16, activation='relu', kernel_regularizer=l1_l2(l1_reg/10, l2_reg/10))(discriminator_input)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(8, activation='relu', kernel_regularizer=l1_l2(l1_reg/10, l2_reg/10))(x)
    x = layers.Dropout(0.2)(x)
    discriminator_output = layers.Dense(1, activation='sigmoid', kernel_regularizer=l1_l2(l1_reg/10, l2_reg/10))(x)
    
    discriminator = models.Model(discriminator_input, discriminator_output, name='regular_discriminator')
    
    return encoder, decoder, discriminator