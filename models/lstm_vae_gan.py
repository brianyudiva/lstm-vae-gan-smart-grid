import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l1_l2

"""
1. HIGH CAPACITY input_shape, latent_dim=12)
2. REGULAR (input_shape, latent_dim=8)
3. COMPACT (input_shape, latent_dim=4)
"""

# === SHARED UTILITIES ===
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
    
    x = layers.LSTM(
        16, 
        return_sequences=True,
        kernel_regularizer=l1_l2(l1_reg/10, l2_reg/10),
        recurrent_regularizer=l1_l2(l1_reg/10, l2_reg/10),
        dropout=0.3,
        recurrent_dropout=0.3
    )(discriminator_input)
    
    x = layers.LSTM(
        8, 
        return_sequences=False,
        kernel_regularizer=l1_l2(l1_reg/10, l2_reg/10),
        recurrent_regularizer=l1_l2(l1_reg/10, l2_reg/10),
        dropout=0.3,
        recurrent_dropout=0.3
    )(x)
    
    x = layers.Dense(16, activation='relu', kernel_regularizer=l1_l2(l1_reg/10, l2_reg/10))(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(8, activation='relu', kernel_regularizer=l1_l2(l1_reg/10, l2_reg/10))(x)
    x = layers.Dropout(0.2)(x)
    discriminator_output = layers.Dense(1, activation='sigmoid', kernel_regularizer=l1_l2(l1_reg/10, l2_reg/10))(x)
    
    discriminator = models.Model(discriminator_input, discriminator_output, name='regular_discriminator')
    
    return encoder, decoder, discriminator

def build_lstm_vae_gan_high_capacity(input_shape, latent_dim=12):
    l1_reg = 1e-5
    l2_reg = 1e-4
    dropout_rate = 0.15
    
    # === ENCODER ===
    encoder_input = layers.Input(shape=input_shape, name='input_layer')
    
    x = layers.LSTM(
        64,  # High capacity
        return_sequences=True, 
        kernel_regularizer=l1_l2(l1_reg, l2_reg),
        recurrent_regularizer=l1_l2(l1_reg, l2_reg),
        dropout=dropout_rate,
        recurrent_dropout=dropout_rate
    )(encoder_input)
    
    x = layers.LSTM(
        32, 
        return_sequences=True,
        kernel_regularizer=l1_l2(l1_reg, l2_reg),
        recurrent_regularizer=l1_l2(l1_reg, l2_reg),
        dropout=dropout_rate,
        recurrent_dropout=dropout_rate
    )(x)
    
    x = layers.LSTM(
        16, 
        return_sequences=False, 
        kernel_regularizer=l1_l2(l1_reg, l2_reg),
        recurrent_regularizer=l1_l2(l1_reg, l2_reg),
        dropout=dropout_rate,
        recurrent_dropout=dropout_rate
    )(x)
    
    # Large dense layers for complex pattern recognition
    x = layers.Dense(64, activation='relu', kernel_regularizer=l1_l2(l1_reg, l2_reg))(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(32, activation='relu', kernel_regularizer=l1_l2(l1_reg, l2_reg))(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(16, activation='relu', kernel_regularizer=l1_l2(l1_reg, l2_reg))(x)
    
    # VAE outputs with larger latent space
    z_mean = layers.Dense(latent_dim, name='z_mean', kernel_regularizer=l1_l2(l1_reg, l2_reg))(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var', kernel_regularizer=l1_l2(l1_reg, l2_reg))(x)
    z = sampling_layer(z_mean, z_log_var, 'sampling')
    
    encoder = models.Model(encoder_input, [z_mean, z_log_var, z], name='high_capacity_encoder')
    
    # === DECODER ===
    decoder_input = layers.Input(shape=(latent_dim,))
    
    # Large decoding path
    x = layers.Dense(16, activation='relu', kernel_regularizer=l1_l2(l1_reg, l2_reg))(decoder_input)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(32, activation='relu', kernel_regularizer=l1_l2(l1_reg, l2_reg))(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=l1_l2(l1_reg, l2_reg))(x)
    x = layers.Dense(32, activation='relu', kernel_regularizer=l1_l2(l1_reg, l2_reg))(x)
    x = layers.RepeatVector(input_shape[0])(x)
    
    # Deep LSTM decoder
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
    
    x = layers.LSTM(
        64, 
        return_sequences=True,
        kernel_regularizer=l1_l2(l1_reg, l2_reg),
        recurrent_regularizer=l1_l2(l1_reg, l2_reg),
        dropout=dropout_rate,
        recurrent_dropout=dropout_rate
    )(x)
    
    decoder_output = layers.Dense(input_shape[1], kernel_regularizer=l1_l2(l1_reg, l2_reg))(x)
    decoder = models.Model(decoder_input, decoder_output, name='high_capacity_decoder')
    
    # === DISCRIMINATOR ===
    discriminator_input = layers.Input(shape=input_shape)
    
    x = layers.LSTM(
        32, 
        return_sequences=True,
        kernel_regularizer=l1_l2(l1_reg/10, l2_reg/10),
        recurrent_regularizer=l1_l2(l1_reg/10, l2_reg/10),
        dropout=0.2,
        recurrent_dropout=0.2
    )(discriminator_input)
    
    x = layers.LSTM(
        16, 
        return_sequences=False,
        kernel_regularizer=l1_l2(l1_reg/10, l2_reg/10),
        recurrent_regularizer=l1_l2(l1_reg/10, l2_reg/10),
        dropout=0.2,
        recurrent_dropout=0.2
    )(x)
    
    x = layers.Dense(32, activation='relu', kernel_regularizer=l1_l2(l1_reg/10, l2_reg/10))(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(16, activation='relu', kernel_regularizer=l1_l2(l1_reg/10, l2_reg/10))(x)
    x = layers.Dropout(0.1)(x)
    discriminator_output = layers.Dense(1, activation='sigmoid', kernel_regularizer=l1_l2(l1_reg/10, l2_reg/10))(x)
    
    discriminator = models.Model(discriminator_input, discriminator_output, name='high_capacity_discriminator')
    
    return encoder, decoder, discriminator

def build_lstm_vae_gan_compact(input_shape, latent_dim=4):    
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

def select_architecture(normal_samples_count, input_shape, latent_dim=8, force_architecture=None):    
    print(f"Building architecture ({normal_samples_count} samples, input shape: {input_shape})")
    
    # Determine architecture based on input complexity and force parameter
    if force_architecture == 'compact':
        encoder, decoder, discriminator = build_lstm_vae_gan_compact(input_shape, latent_dim)
        arch_name = 'compact'
    elif force_architecture == 'regular':
        encoder, decoder, discriminator = build_lstm_vae_gan_regular(input_shape, latent_dim)
        arch_name = 'regular'
    # elif force_architecture == 'high_capacity':
    #     encoder, decoder, discriminator = build_lstm_vae_gan_high_capacity(input_shape, latent_dim)
    #     arch_name = 'high_capacity'
    else:
        # Auto-select based on input dimensionality
        input_complexity = input_shape[0] * input_shape[1]  # sequence_len * features
        
        # if input_complexity > 600:  # High-dimensional like Jacobian dataset
        #     encoder, decoder, discriminator = build_lstm_vae_gan_high_capacity(input_shape, latent_dim)
        #     arch_name = 'high_capacity'
        #     print(f"   Auto-selected HIGH CAPACITY architecture for input ({input_complexity} dims)")
        if input_complexity > 200:
            encoder, decoder, discriminator = build_lstm_vae_gan_regular(input_shape, latent_dim)
            arch_name = 'regular'
            print(f"   Auto-selected REGULAR architecture for input ({input_complexity} dims)")
        else:
            encoder, decoder, discriminator = build_lstm_vae_gan_compact(input_shape, latent_dim)
            arch_name = 'compact'
            print(f"   Auto-selected COMPACT architecture for input ({input_complexity} dims)")
    
    # Calculate parameters
    total_params = encoder.count_params() + decoder.count_params()
    if discriminator:
        total_params += discriminator.count_params()
    
    data_param_ratio = normal_samples_count / total_params
    
    print(f"Architecture stats:")
    print(f"   Architecture: {arch_name.upper()}")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Data/Parameter ratio: {data_param_ratio:.2f}")
    print(f"   Latent dimension: {latent_dim}")
    
    if arch_name == 'regular':
        print(f"   Regular model")
    else:
        print(f"   Compact model")
    
    return encoder, decoder, discriminator