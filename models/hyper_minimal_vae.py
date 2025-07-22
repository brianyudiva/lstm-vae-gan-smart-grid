import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l1_l2


def build_hyper_minimal_vae(input_shape, latent_dim=1):
    """Hyper-minimal VAE with <200 parameters for extreme overfitting scenarios
    
    Target: 1489 samples with <200 parameters (ratio > 7.4)
    Current issue: 1,442 parameters vs 1,489 samples (ratio: 1.03)
    """
    
    # === Encoder: Ultra-minimal compression ===
    encoder_input = layers.Input(shape=input_shape, name='input_layer')
    
    # Flatten and compress in one step
    x = layers.Flatten()(encoder_input)  # (12, 18) -> 216 features
    
    # Single bottleneck layer - extreme compression
    x = layers.Dense(3,  # Only 3 hidden units
                     activation='tanh',
                     kernel_regularizer=l1_l2(l1=0.01, l2=0.01),
                     name='bottleneck')(x)
    x = layers.Dropout(0.5)(x)
    
    # VAE outputs - single dimension latent space
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
    
    # Sampling function
    def sampling(args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    z = layers.Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    encoder = models.Model(encoder_input, [z_mean, z_log_var, z], name="hyper_minimal_encoder")
    
    # === Decoder: Ultra-minimal reconstruction ===
    decoder_input = layers.Input(shape=(latent_dim,))
    
    # Expand through bottleneck
    x = layers.Dense(3,  # Same bottleneck size
                     activation='tanh',
                     kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(decoder_input)
    x = layers.Dropout(0.5)(x)
    
    # Direct to output size
    x = layers.Dense(input_shape[0] * input_shape[1],  # 216 features
                     kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(x)
    
    # Reshape to sequence
    decoder_output = layers.Reshape(input_shape)(x)
    
    decoder = models.Model(decoder_input, decoder_output, name="hyper_minimal_decoder")
    
    # Print parameter count
    encoder_params = sum([tf.size(v).numpy() for v in encoder.trainable_weights])
    decoder_params = sum([tf.size(v).numpy() for v in decoder.trainable_weights])
    total_params = encoder_params + decoder_params
    
    print(f"🔢 HYPER-MINIMAL MODEL PARAMETERS:")
    print(f"   Encoder: {encoder_params} parameters")
    print(f"   Decoder: {decoder_params} parameters") 
    print(f"   Total: {total_params} parameters")
    print(f"   Target: <200 parameters for ratio >7.4")
    
    # No discriminator - pure VAE approach
    discriminator = None
    
    return encoder, decoder, discriminator


def build_extreme_minimal_vae(input_shape, latent_dim=1):
    """Even more extreme - target <100 parameters"""
    
    # === Encoder ===
    encoder_input = layers.Input(shape=input_shape, name='input_layer')
    x = layers.Flatten()(encoder_input)
    
    # No hidden layer - direct compression
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
    
    def sampling(args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    z = layers.Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    encoder = models.Model(encoder_input, [z_mean, z_log_var, z], name="extreme_minimal_encoder")
    
    # === Decoder ===
    decoder_input = layers.Input(shape=(latent_dim,))
    
    # Direct expansion - no hidden layers
    x = layers.Dense(input_shape[0] * input_shape[1])(decoder_input)
    decoder_output = layers.Reshape(input_shape)(x)
    
    decoder = models.Model(decoder_input, decoder_output, name="extreme_minimal_decoder")
    
    # Print parameter count
    encoder_params = sum([tf.size(v).numpy() for v in encoder.trainable_weights])
    decoder_params = sum([tf.size(v).numpy() for v in decoder.trainable_weights])
    total_params = encoder_params + decoder_params
    
    print(f"🔢 EXTREME MINIMAL MODEL PARAMETERS:")
    print(f"   Encoder: {encoder_params} parameters")
    print(f"   Decoder: {decoder_params} parameters") 
    print(f"   Total: {total_params} parameters")
    print(f"   Expected: ~{(216+1)*latent_dim*2} parameters")
    
    return encoder, decoder, None
