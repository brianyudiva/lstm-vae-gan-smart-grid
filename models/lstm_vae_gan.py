import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l1_l2
import numpy as np

# === SHARED UTILITIES ===
def compute_fft_features(input_tensor):
    """
    Compute FFT features for temporal pattern detection with fixed output shape
    Extract 4-band power ratios per feature → normalize → return fixed-size vector
    
    Args:
        input_tensor: Input tensor of shape (batch, timesteps, features)
        
    Returns:
        FFT magnitude features with exactly 16 components (4 bands × 4 summary features)
    """
    # Apply FFT to each feature separately
    fft_real = tf.signal.rfft(input_tensor)  # Real FFT to avoid complex numbers
    fft_magnitude = tf.abs(fft_real)  # Get magnitude
    
    # Get the shape information
    batch_size = tf.shape(fft_magnitude)[0]
    fft_timesteps = tf.shape(fft_magnitude)[1]  # This will be (timesteps//2 + 1)
    n_features = tf.shape(fft_magnitude)[2]
    
    # Define 4 frequency bands for each feature
    # Band 1: Low frequency (0-25% of Nyquist)
    # Band 2: Low-mid frequency (25-50% of Nyquist) 
    # Band 3: High-mid frequency (50-75% of Nyquist)
    # Band 4: High frequency (75-100% of Nyquist)
    
    band_1_end = tf.maximum(1, fft_timesteps // 4)
    band_2_end = tf.maximum(2, fft_timesteps // 2)
    band_3_end = tf.maximum(3, 3 * fft_timesteps // 4)
    
    # Extract power in each band for each feature
    band_1_power = tf.reduce_sum(tf.square(fft_magnitude[:, :band_1_end, :]), axis=1)  # (batch, features)
    band_2_power = tf.reduce_sum(tf.square(fft_magnitude[:, band_1_end:band_2_end, :]), axis=1)
    band_3_power = tf.reduce_sum(tf.square(fft_magnitude[:, band_2_end:band_3_end, :]), axis=1)
    band_4_power = tf.reduce_sum(tf.square(fft_magnitude[:, band_3_end:, :]), axis=1)
    
    # Calculate total power per feature for normalization
    total_power = band_1_power + band_2_power + band_3_power + band_4_power + 1e-8
    
    # Compute power ratios (normalized)
    ratio_1 = band_1_power / total_power
    ratio_2 = band_2_power / total_power
    ratio_3 = band_3_power / total_power
    ratio_4 = band_4_power / total_power
    
    # Summary statistics across all features (4 values)
    summary_features = tf.stack([
        tf.reduce_mean(ratio_1, axis=1),  # Mean low-freq ratio across features
        tf.reduce_mean(ratio_2, axis=1),  # Mean low-mid-freq ratio
        tf.reduce_mean(ratio_3, axis=1),  # Mean high-mid-freq ratio  
        tf.reduce_mean(ratio_4, axis=1)   # Mean high-freq ratio
    ], axis=1)  # Shape: (batch, 4)
    
    # Additional summary features (12 more values for total of 16)
    additional_features = tf.stack([
        tf.math.reduce_std(ratio_1, axis=1),   # Std of low-freq ratios
        tf.math.reduce_std(ratio_2, axis=1),   # Std of low-mid-freq ratios
        tf.math.reduce_std(ratio_3, axis=1),   # Std of high-mid-freq ratios
        tf.math.reduce_std(ratio_4, axis=1),   # Std of high-freq ratios
        tf.reduce_max(ratio_1, axis=1),   # Max low-freq ratio
        tf.reduce_max(ratio_2, axis=1),   # Max low-mid-freq ratio
        tf.reduce_max(ratio_3, axis=1),   # Max high-mid-freq ratio
        tf.reduce_max(ratio_4, axis=1),   # Max high-freq ratio
        tf.reduce_min(ratio_1, axis=1),   # Min low-freq ratio
        tf.reduce_min(ratio_2, axis=1),   # Min low-mid-freq ratio  
        tf.reduce_min(ratio_3, axis=1),   # Min high-mid-freq ratio
        tf.reduce_min(ratio_4, axis=1),   # Min high-freq ratio
    ], axis=1)  # Shape: (batch, 12)
    
    # Combine all features (4 + 12 = 16 total)
    fft_features_fixed = tf.concat([summary_features, additional_features], axis=1)
    
    # Final normalization to ensure stable training
    epsilon = 1e-8
    fft_norm = tf.norm(fft_features_fixed, axis=1, keepdims=True) + epsilon
    fft_normalized = fft_features_fixed / fft_norm
    
    return fft_normalized

def enhanced_discriminator_block(input_tensor, l1_reg, l2_reg, dropout_rate, name_prefix):
    """
    Enhanced discriminator block with FFT analysis and extra bidirectional LSTM layer
    
    Args:
        input_tensor: Input tensor of shape (batch, timesteps, features)
        l1_reg, l2_reg: Regularization parameters
        dropout_rate: Dropout rate
        name_prefix: Prefix for layer names
        
    Returns:
        Enhanced discriminator output
    """
    # === TIME-DOMAIN PROCESSING ===
    # First LSTM layer
    lstm1 = layers.LSTM(
        16, 
        return_sequences=True,
        kernel_regularizer=l1_l2(l1_reg/10, l2_reg/10),
        recurrent_regularizer=l1_l2(l1_reg/10, l2_reg/10),
        dropout=dropout_rate,
        recurrent_dropout=dropout_rate,
        name=f'{name_prefix}_lstm1'
    )(input_tensor)
    
    # Second LSTM layer (extra layer for better temporal pattern detection)
    lstm2 = layers.LSTM(
        16,  # Same hidden size as first layer
        return_sequences=True,
        kernel_regularizer=l1_l2(l1_reg/10, l2_reg/10),
        recurrent_regularizer=l1_l2(l1_reg/10, l2_reg/10),
        dropout=dropout_rate,
        recurrent_dropout=dropout_rate,
        name=f'{name_prefix}_lstm2'
    )(lstm1)
    
    # Extra bidirectional LSTM layer for stealth sensitivity
    bidirectional_lstm = layers.Bidirectional(
        layers.LSTM(
            16,  # Same hidden size
            return_sequences=False,
            kernel_regularizer=l1_l2(l1_reg/10, l2_reg/10),
            recurrent_regularizer=l1_l2(l1_reg/10, l2_reg/10),
            dropout=dropout_rate,
            recurrent_dropout=dropout_rate
        ),
        name=f'{name_prefix}_bidirectional_lstm'
    )(lstm2)
    
    # Final LSTM layer (output single vector)
    lstm_output = layers.LSTM(
        8, 
        return_sequences=False,
        kernel_regularizer=l1_l2(l1_reg/10, l2_reg/10),
        recurrent_regularizer=l1_l2(l1_reg/10, l2_reg/10),
        dropout=dropout_rate,
        recurrent_dropout=dropout_rate,
        name=f'{name_prefix}_lstm_final'
    )(lstm2)
    
    # === FREQUENCY-DOMAIN PROCESSING ===
    # Compute FFT features with 4-band power ratios
    fft_features = layers.Lambda(
        compute_fft_features,
        output_shape=(16,),  # Fixed 16 FFT features (4 bands × 4 summary stats)
        name=f'{name_prefix}_fft_features'
    )(input_tensor)
    
    # Dense layer for FFT feature processing
    fft_processed = layers.Dense(
        16,  # Process 16 FFT features
        activation='relu',
        kernel_regularizer=l1_l2(l1_reg/10, l2_reg/10),
        name=f'{name_prefix}_fft_dense'
    )(fft_features)
    fft_processed = layers.Dropout(dropout_rate/2, name=f'{name_prefix}_fft_dropout')(fft_processed)
    
    # === FEATURE FUSION ===
    # Concatenate time-domain features (both regular and bidirectional) and frequency-domain features
    combined_features = layers.Concatenate(name=f'{name_prefix}_feature_fusion')([
        lstm_output,        # Regular LSTM output (8 features)
        bidirectional_lstm, # Bidirectional LSTM output (32 features: 16*2)
        fft_processed       # Processed FFT features (16 features)
    ])
    
    # Additional dense layer before final classifier (as requested)
    enhanced_features = layers.Dense(
        64,
        activation='relu',
        kernel_regularizer=l1_l2(l1_reg/10, l2_reg/10),
        name=f'{name_prefix}_enhanced_dense'
    )(combined_features)
    enhanced_features = layers.Dropout(dropout_rate/2, name=f'{name_prefix}_enhanced_dropout')(enhanced_features)
    
    return enhanced_features

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
    
    # Multi-layer LSTM with increasing capacity
    x = layers.LSTM(
        32,
        return_sequences=True, 
        kernel_regularizer=l1_l2(l1_reg, l2_reg),
        recurrent_regularizer=l1_l2(l1_reg, l2_reg),
        dropout=dropout_rate,
        recurrent_dropout=dropout_rate
    )(encoder_input)
    
    x = layers.LSTM(
        16, 
        return_sequences=False, 
        kernel_regularizer=l1_l2(l1_reg, l2_reg),
        recurrent_regularizer=l1_l2(l1_reg, l2_reg),
        dropout=dropout_rate,
        recurrent_dropout=dropout_rate
    )(x)
    
    # Larger dense layers for high-dimensional mapping
    x = layers.Dense(32, activation='relu', kernel_regularizer=l1_l2(l1_reg, l2_reg))(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(16, activation='relu', kernel_regularizer=l1_l2(l1_reg, l2_reg))(x)
    x = layers.Dropout(0.2)(x)
    
    # VAE outputs
    z_mean = layers.Dense(latent_dim, name='z_mean', kernel_regularizer=l1_l2(l1_reg, l2_reg))(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var', kernel_regularizer=l1_l2(l1_reg, l2_reg))(x)
    z = sampling_layer(z_mean, z_log_var, 'sampling')
    
    encoder = models.Model(encoder_input, [z_mean, z_log_var, z], name='regular_encoder')
    
    # === DECODER ===
    decoder_input = layers.Input(shape=(latent_dim,))
    
    # Larger decoding path
    x = layers.Dense(16, activation='relu', kernel_regularizer=l1_l2(l1_reg, l2_reg))(decoder_input)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation='relu', kernel_regularizer=l1_l2(l1_reg, l2_reg))(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(16, activation='relu', kernel_regularizer=l1_l2(l1_reg, l2_reg))(x)
    x = layers.RepeatVector(input_shape[0])(x)
    
    # Multi-layer LSTM decoder
    x = layers.LSTM(
        16, 
        return_sequences=True,
        kernel_regularizer=l1_l2(l1_reg, l2_reg),
        recurrent_regularizer=l1_l2(l1_reg, l2_reg),
        dropout=dropout_rate,
        recurrent_dropout=dropout_rate
    )(x)
    
    x = layers.LSTM(
        32, 
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
    
    # Use enhanced discriminator block with FFT analysis
    enhanced_features = enhanced_discriminator_block(
        discriminator_input, 
        l1_reg, 
        l2_reg, 
        0.3,  # dropout_rate
        'regular_disc'
    )
    
    # Final classification layers
    x = layers.Dense(16, activation='relu', kernel_regularizer=l1_l2(l1_reg/10, l2_reg/10))(enhanced_features)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(8, activation='relu', kernel_regularizer=l1_l2(l1_reg/10, l2_reg/10))(x)
    x = layers.Dropout(0.2)(x)
    discriminator_output = layers.Dense(1, activation='sigmoid', kernel_regularizer=l1_l2(l1_reg/10, l2_reg/10))(x)
    
    discriminator = models.Model(discriminator_input, discriminator_output, name='regular_discriminator')
    
    return encoder, decoder, discriminator