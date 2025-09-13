"""
Loss functions for LSTM-VAE-GAN anomaly detection
"""
import tensorflow as tf
import numpy as np


def kl_loss(z_mean, z_log_var):
    """KL divergence loss for VAE"""
    return -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))


def reconstruction_loss(y_true, y_pred):
    """Standard MSE reconstruction loss"""
    return tf.reduce_mean(tf.square(y_true - y_pred))

def robust_reconstruction_loss(y_true, y_pred, epsilon=0.01):
    """Simplified robust loss (Huber-style)"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    diff = y_true - y_pred
    squared_diff = tf.square(diff)
    abs_diff = tf.abs(diff)
    
    # Huber loss
    huber_loss = tf.where(
        abs_diff <= epsilon,
        0.5 * squared_diff,
        epsilon * abs_diff - 0.5 * epsilon**2
    )
    
    return tf.reduce_mean(huber_loss)

def regularization_loss(encoder, decoder):
    """Enhanced regularization for better generalization"""
    l2_loss = 0
    for layer in encoder.layers:
        if hasattr(layer, 'kernel'):
            l2_loss += tf.reduce_sum(tf.square(layer.kernel))
        if hasattr(layer, 'bias') and layer.bias is not None:
            l2_loss += tf.reduce_sum(tf.square(layer.bias))
    
    for layer in decoder.layers:
        if hasattr(layer, 'kernel'):
            l2_loss += tf.reduce_sum(tf.square(layer.kernel))
        if hasattr(layer, 'bias') and layer.bias is not None:
            l2_loss += tf.reduce_sum(tf.square(layer.bias))
    
    return l2_loss * 0.01

def beta_schedule(epoch, total_epochs, max_beta=1.0, warmup_epochs=8):
    """Balanced beta scheduling for effective anomaly detection"""
    if epoch < warmup_epochs:
        return 0.0001 + (max_beta - 0.0001) * (epoch / warmup_epochs) ** 2 
    else:
        return max_beta

def vae_loss(y_true, y_pred, encoder, latent_dim, beta=1.0):
    """Combined VAE loss function for baseline models"""
    recon_loss = reconstruction_loss(y_true, y_pred)
    
    encoder_output = encoder(y_true)
    if isinstance(encoder_output, list) and len(encoder_output) >= 2:
        z_mean, z_log_var = encoder_output[0], encoder_output[1]
    else:
        batch_size = tf.shape(y_true)[0]
        z_mean = tf.zeros((batch_size, latent_dim))
        z_log_var = tf.zeros((batch_size, latent_dim))
    
    kl_loss_val = kl_loss(z_mean, z_log_var)
    
    return recon_loss + beta * kl_loss_val
