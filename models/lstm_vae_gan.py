import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l1_l2


# === SHARED UTILITIES ===
def sampling_layer(z_mean, z_log_var, name='sampling'):
    """Standardized reparameterization trick for VAE"""
    def sampling(args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    # Get latent dimension from z_mean shape at build time
    latent_dim = z_mean.shape[-1]
    return layers.Lambda(sampling, output_shape=(latent_dim,), name=name)([z_mean, z_log_var])


# === REGULAR LSTM-VAE-GAN (for sufficient training data) ===
def build_encoder(input_shape, latent_dim, add_batch_norm=True):
    """Standard encoder with optional batch normalization"""
    inputs = tf.keras.Input(shape=input_shape)
    
    x = layers.LSTM(64, return_sequences=True)(inputs)
    if add_batch_norm:
        x = layers.BatchNormalization()(x)
    
    x = layers.LSTM(32)(x)
    if add_batch_norm:
        x = layers.BatchNormalization()(x)
    
    # VAE: Output mean and log variance
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
    
    # Reparameterization trick
    z = sampling_layer(z_mean, z_log_var, 'z')
    
    return tf.keras.Model(inputs, [z_mean, z_log_var, z], name="Encoder")

def build_decoder(sequence_length, feature_dim, latent_dim, add_batch_norm=True):
    """Standard decoder with optional batch normalization"""
    inputs = tf.keras.Input(shape=(latent_dim,))
    x = layers.RepeatVector(sequence_length)(inputs)
    
    x = layers.LSTM(32, return_sequences=True)(x)
    if add_batch_norm:
        x = layers.BatchNormalization()(x)
        
    x = layers.LSTM(64, return_sequences=True)(x)
    if add_batch_norm:
        x = layers.BatchNormalization()(x)
        
    outputs = layers.TimeDistributed(layers.Dense(feature_dim))(x)
    return tf.keras.Model(inputs, outputs, name="Decoder")


def build_discriminator(sequence_length, feature_dim, add_batch_norm=True):
    """Standard discriminator with optional batch normalization"""
    inputs = tf.keras.Input(shape=(sequence_length, feature_dim))
    
    x = layers.LSTM(64, return_sequences=True)(inputs)
    if add_batch_norm:
        x = layers.BatchNormalization()(x)
        
    x = layers.LSTM(32)(x)
    if add_batch_norm:
        x = layers.BatchNormalization()(x)
        
    x = layers.Dense(1, activation='sigmoid')(x)
    return tf.keras.Model(inputs, x, name="Discriminator")


def build_lstm_vae_gan(input_shape, latent_dim, add_batch_norm=True):
    """
    Standard LSTM-VAE-GAN architecture
    
    Use when:
    - You have sufficient training data (>5000 normal samples)
    - Data/parameter ratio > 10
    - Want best reconstruction quality
    """
    encoder = build_encoder(input_shape, latent_dim, add_batch_norm)
    decoder = build_decoder(input_shape[0], input_shape[1], latent_dim, add_batch_norm)
    discriminator = build_discriminator(input_shape[0], input_shape[1], add_batch_norm)
    return encoder, decoder, discriminator

# === COMPACT ARCHITECTURE (for limited data) ===
def build_lstm_vae_gan_compact(input_shape, latent_dim=4):
    """
    Compact LSTM-VAE-GAN with regularization for anomaly detection
    
    Use when:
    - Limited training data (1000-5000 normal samples)
    - Data/parameter ratio 3-10
    - Need balance between quality and overfitting prevention
    """
    
    # Regularization parameters
    l1_reg = 1e-3
    l2_reg = 1e-2
    dropout_rate = 0.7
    
    # === ENCODER ===
    encoder_input = layers.Input(shape=input_shape, name='input_layer')
    
    x = layers.LSTM(
        8, 
        return_sequences=True, 
        kernel_regularizer=l1_l2(l1_reg, l2_reg),
        recurrent_regularizer=l1_l2(l1_reg, l2_reg),
        dropout=dropout_rate,
        recurrent_dropout=dropout_rate
    )(encoder_input)
    
    x = layers.LSTM(
        4, 
        return_sequences=False, 
        kernel_regularizer=l1_l2(l1_reg, l2_reg),
        recurrent_regularizer=l1_l2(l1_reg, l2_reg),
        dropout=dropout_rate,
        recurrent_dropout=dropout_rate
    )(x)
    
    x = layers.Dense(8, activation='relu', kernel_regularizer=l1_l2(l1_reg, l2_reg))(x)
    x = layers.Dropout(0.5)(x)
    
    # VAE outputs
    z_mean = layers.Dense(latent_dim, name='z_mean', kernel_regularizer=l1_l2(l1_reg, l2_reg))(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var', kernel_regularizer=l1_l2(l1_reg, l2_reg))(x)
    z = sampling_layer(z_mean, z_log_var, 'sampling')
    
    encoder = models.Model(encoder_input, [z_mean, z_log_var, z], name='compact_encoder')
    
    # === DECODER ===
    decoder_input = layers.Input(shape=(latent_dim,))
    x = layers.Dense(8, activation='relu', kernel_regularizer=l1_l2(l1_reg, l2_reg))(decoder_input)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(4, activation='relu', kernel_regularizer=l1_l2(l1_reg, l2_reg))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.RepeatVector(input_shape[0])(x)
    
    x = layers.LSTM(
        4, 
        return_sequences=True,
        kernel_regularizer=l1_l2(l1_reg, l2_reg),
        recurrent_regularizer=l1_l2(l1_reg, l2_reg),
        dropout=dropout_rate,
        recurrent_dropout=dropout_rate
    )(x)
    
    decoder_output = layers.Dense(input_shape[1], kernel_regularizer=l1_l2(l1_reg, l2_reg))(x)
    decoder = models.Model(decoder_input, decoder_output, name='compact_decoder')
    
    # === DISCRIMINATOR ===
    discriminator_input = layers.Input(shape=input_shape)
    x = layers.LSTM(
        4, 
        return_sequences=False,
        kernel_regularizer=l1_l2(l1_reg/10, l2_reg/10),
        recurrent_regularizer=l1_l2(l1_reg/10, l2_reg/10),
        dropout=0.5,
        recurrent_dropout=0.5
    )(discriminator_input)
    
    x = layers.Dense(8, activation='relu', kernel_regularizer=l1_l2(l1_reg/10, l2_reg/10))(x)
    x = layers.Dropout(0.3)(x)
    discriminator_output = layers.Dense(1, activation='sigmoid', kernel_regularizer=l1_l2(l1_reg/10, l2_reg/10))(x)
    
    discriminator = models.Model(discriminator_input, discriminator_output, name='compact_discriminator')
    
    return encoder, decoder, discriminator

# === ULTRA-COMPACT ARCHITECTURE (for very limited data) ===
def build_ultra_compact_lstm_vae_gan(input_shape, latent_dim=2):
    """
    Ultra-compact LSTM-VAE-GAN with aggressive regularization
    
    Use when:
    - Very limited training data (<1000 normal samples)
    - Data/parameter ratio < 3
    - Severe overfitting risk
    - Research/experimental purposes
    """
    
    # Very strong regularization
    l1_reg = 1e-2
    l2_reg = 1e-1
    dropout_rate = 0.8
    
    # === ENCODER ===
    encoder_input = layers.Input(shape=input_shape, name='input_layer')
    
    x = layers.LSTM(
        4,
        return_sequences=False, 
        kernel_regularizer=l1_l2(l1_reg, l2_reg),
        recurrent_regularizer=l1_l2(l1_reg, l2_reg),
        dropout=dropout_rate,
        recurrent_dropout=dropout_rate
    )(encoder_input)
    
    x = layers.Dropout(0.7)(x)
    
    # VAE outputs
    z_mean = layers.Dense(latent_dim, name='z_mean', kernel_regularizer=l1_l2(l1_reg, l2_reg))(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var', kernel_regularizer=l1_l2(l1_reg, l2_reg))(x)
    z = sampling_layer(z_mean, z_log_var, 'sampling')
    
    encoder = models.Model(encoder_input, [z_mean, z_log_var, z], name='ultra_compact_encoder')
    
    # === DECODER ===
    decoder_input = layers.Input(shape=(latent_dim,))
    x = layers.Dense(4, activation='relu', kernel_regularizer=l1_l2(l1_reg, l2_reg))(decoder_input)
    x = layers.Dropout(0.7)(x)
    x = layers.RepeatVector(input_shape[0])(x)
    
    x = layers.LSTM(
        4, 
        return_sequences=True,
        kernel_regularizer=l1_l2(l1_reg, l2_reg),
        recurrent_regularizer=l1_l2(l1_reg, l2_reg),
        dropout=dropout_rate,
        recurrent_dropout=dropout_rate
    )(x)
    
    decoder_output = layers.Dense(input_shape[1], kernel_regularizer=l1_l2(l1_reg, l2_reg))(x)
    decoder = models.Model(decoder_input, decoder_output, name='ultra_compact_decoder')
    
    # No discriminator - too complex for ultra-compact case
    return encoder, decoder, None


# === MODEL SELECTION HELPER ===
def select_architecture(normal_samples_count, input_shape, latent_dim=8):
    """
    Automatically select the best architecture based on data size
    
    Args:
        normal_samples_count: Number of normal training samples
        input_shape: Shape of input sequences (seq_length, features)
        latent_dim: Latent space dimension
        
    Returns:
        encoder, decoder, discriminator, architecture_info
    """
    
    if normal_samples_count >= 5000:
        print(f"🏗️ Selected: REGULAR architecture ({normal_samples_count} samples)")
        encoder, decoder, discriminator = build_lstm_vae_gan(input_shape, latent_dim)
        info = {
            'name': 'regular',
            'recommended_lr': 2e-4,
            'kl_weight': 1.0,
            'recon_weight': 100.0,
            'adv_weight': 1.0
        }
        
    elif normal_samples_count >= 1000:
        print(f"🏗️ Selected: COMPACT architecture ({normal_samples_count} samples)")
        encoder, decoder, discriminator = build_lstm_vae_gan_compact(input_shape, latent_dim)
        info = {
            'name': 'compact',
            'recommended_lr': 1e-4,
            'kl_weight': 10.0,
            'recon_weight': 1000.0,
            'adv_weight': 0.1
        }
        
    else:
        print(f"🏗️ Selected: ULTRA-COMPACT architecture ({normal_samples_count} samples)")
        encoder, decoder, discriminator = build_ultra_compact_lstm_vae_gan(input_shape, latent_dim//2)
        info = {
            'name': 'ultra_compact',
            'recommended_lr': 5e-5,
            'kl_weight': 50.0,
            'recon_weight': 10000.0,
            'adv_weight': 0.0
        }
    
    # Calculate parameters
    total_params = encoder.count_params() + decoder.count_params()
    if discriminator:
        total_params += discriminator.count_params()
    
    data_param_ratio = normal_samples_count / total_params
    
    print(f"📊 Architecture stats:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Data/Parameter ratio: {data_param_ratio:.2f}")
    print(f"   Has discriminator: {discriminator is not None}")
    
    if data_param_ratio < 3:
        print("⚠️ WARNING: Very low data/parameter ratio - high overfitting risk")
    elif data_param_ratio < 10:
        print("⚠️ CAUTION: Low data/parameter ratio - monitor for overfitting")
    else:
        print("✅ GOOD: Sufficient data/parameter ratio")
    
    return encoder, decoder, discriminator, info
