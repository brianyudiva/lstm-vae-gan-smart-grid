import numpy as np
import tensorflow as tf
from models.lstm_vae_gan import select_architecture
import os
from sklearn.metrics import classification_report, confusion_matrix

def kl_loss(z_mean, z_log_var):
    return -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))

def reconstruction_loss(y_true, y_pred):
    """Standard MSE reconstruction loss for regular model"""
    return tf.reduce_mean(tf.square(y_true - y_pred))

def compute_anomaly_scores(encoder, decoder, X_test, X_train_normal, threshold_percentile=95):
    """Compute anomaly scores with proper threshold from training data"""
    # Get test reconstruction errors
    z_mean, z_log_var, z = encoder(X_test)
    reconstructed = decoder(z)
    test_recon_errors = tf.reduce_mean(tf.square(X_test - reconstructed), axis=[1, 2])
    
    # Get training reconstruction errors (NORMAL DATA ONLY)
    z_mean_train, z_log_var_train, z_train = encoder(X_train_normal)
    reconstructed_train = decoder(z_train)
    train_recon_errors = tf.reduce_mean(tf.square(X_train_normal - reconstructed_train), axis=[1, 2])
    
    # Set threshold based on TRAINING data
    threshold = np.percentile(train_recon_errors, threshold_percentile)
    
    # Binary predictions
    predictions = (test_recon_errors > threshold).numpy().astype(int)
    
    return test_recon_errors.numpy(), predictions, threshold

def evaluate_with_labels(encoder, decoder, X_test, y_test_labels):
    """Evaluate against actual FDIA labels"""
    recon_errors, _, _ = compute_anomaly_scores(encoder, decoder, X_test, X_train_normal, threshold_percentile=95)
    
    # Try different thresholds
    thresholds = np.percentile(recon_errors, [90, 95, 97, 99])
    
    print("\n=== EVALUATION WITH ACTUAL LABELS ===")
    for i, thresh in enumerate(thresholds):
        predictions = (recon_errors > thresh).astype(int)
        print(f"\nThreshold (percentile {[90, 95, 97, 99][i]}): {thresh:.4f}")
        print(f"Predicted anomalies: {np.sum(predictions)}/{len(predictions)}")
        
        if len(np.unique(y_test_labels)) > 1:  # If we have both normal and anomaly samples
            print("Classification Report:")
            print(classification_report(y_test_labels, predictions))

# === CONFIG ===
sequence_path = "data/sequences"
output_path = "outputs/checkpoints"
os.makedirs(output_path, exist_ok=True)

# === LOAD DATA ===
X_train = np.load(f"{sequence_path}/X_train.npy")
X_test = np.load(f"{sequence_path}/X_test.npy")

# Load labels
try:
    y_train = np.load(f"{sequence_path}/y_train_binary.npy")
    y_test = np.load(f"{sequence_path}/y_test_binary.npy")
    print(f"Labels loaded - Train FDIA: {np.sum(y_train)}/{len(y_train)}, Test FDIA: {np.sum(y_test)}/{len(y_test)}")
    
    X_train_normal = X_train[y_train == 0]
    print(f"Training on NORMAL data only: {len(X_train_normal)} samples (was {len(X_train)})")
    
except:
    print("No labels found")
    X_train_normal = X_train
    y_train, y_test = None, None

# === INTELLIGENT ARCHITECTURE SELECTION ===
input_shape = (X_train_normal.shape[1], X_train_normal.shape[2])
print(f"Input shape: {input_shape}")

# Automatically select the best architecture based on data size
encoder, decoder, discriminator, arch_info = select_architecture(
    normal_samples_count=len(X_train_normal),
    input_shape=input_shape,
    latent_dim=8  # Will be adjusted automatically for ultra-compact
)

print(f"\n🎯 Selected architecture: {arch_info['name'].upper()}")

# Check if discriminator exists
use_discriminator = discriminator is not None
print(f"Using discriminator: {use_discriminator}")

# === COMPILE MODELS WITH RECOMMENDED HYPERPARAMETERS ===
# Use architecture-specific hyperparameters
initial_lr = arch_info['recommended_lr']
kl_weight = arch_info['kl_weight']
recon_weight = arch_info['recon_weight']
adv_weight = arch_info['adv_weight']

print(f"\n🎛️ TRAINING CONFIGURATION:")
print(f"Learning rate: {initial_lr}")
print(f"Loss weights - KL: {kl_weight}, Recon: {recon_weight}, Adv: {adv_weight}")

generator_optimizer = tf.keras.optimizers.Adam(initial_lr)
if use_discriminator:
    discriminator_optimizer = tf.keras.optimizers.Adam(initial_lr)
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)

# === TRAINING PARAMETERS ===
batch_size = 32
epochs = 100
steps_per_epoch = X_train_normal.shape[0] // batch_size

# Adjust training parameters based on architecture
if arch_info['name'] == 'ultra_compact':
    epochs = 150  # More epochs for smaller model
    patience = 25
elif arch_info['name'] == 'compact':
    epochs = 120
    patience = 20
else:
    epochs = 100
    patience = 15

print(f"Training epochs: {epochs}, Patience: {patience}")
print(f"Batch size: {batch_size}, Steps per epoch: {steps_per_epoch}")

# === MODEL VALIDATION ===
print(f"\n🧪 TESTING MODEL ARCHITECTURE...")
test_batch = X_train_normal[:2]
test_z_mean, test_z_log_var, test_z = encoder(test_batch)
test_reconstructed = decoder(test_z)

if use_discriminator:
    test_disc_output = discriminator(test_batch)
    print(f"✅ Model test successful - shapes: z={test_z.shape}, recon={test_reconstructed.shape}, disc={test_disc_output.shape}")
else:
    print(f"✅ Model test successful - shapes: z={test_z.shape}, recon={test_reconstructed.shape}")
    print("🎯 Training as VAE only (no discriminator)")

# === TRAINING LOOP ===
print(f"\n🏋️ STARTING TRAINING...")
best_recon_loss = float('inf')
wait = 0

for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    epoch_losses = {'d_loss': 0, 'g_loss': 0, 'kl_loss': 0, 'recon_loss': 0}

    for step in range(steps_per_epoch):
        idx = np.random.randint(0, X_train_normal.shape[0], batch_size)
        real_seq = X_train_normal[idx]

        # === Train Discriminator ===
        if use_discriminator:
            discriminator.trainable = True
            
            z_mean, z_log_var, z = encoder(real_seq)
            fake_seq = decoder(z)

            with tf.GradientTape() as tape:
                real_pred = discriminator(real_seq)
                fake_pred = discriminator(fake_seq)
                d_loss = (bce(tf.ones_like(real_pred), real_pred) + 
                         bce(tf.zeros_like(fake_pred), fake_pred))

            d_grads = tape.gradient(d_loss, discriminator.trainable_weights)
            if d_grads and all(g is not None for g in d_grads):
                discriminator_optimizer.apply_gradients(zip(d_grads, discriminator.trainable_weights))
        else:
            d_loss = tf.constant(0.0)  # Dummy value for logging

        # === Train Generator ===
        if use_discriminator:
            discriminator.trainable = False

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = encoder(real_seq)
            reconstructed = decoder(z)
            
            kl_loss_val = kl_loss(z_mean, z_log_var)
            recon_loss_val = reconstruction_loss(real_seq, reconstructed)
            
            if use_discriminator:
                fake_pred = discriminator(reconstructed)
                adv_loss_val = bce(tf.ones_like(fake_pred), fake_pred)
            else:
                adv_loss_val = tf.constant(0.0)
            
            g_loss = (kl_weight * kl_loss_val + 
                     recon_weight * recon_loss_val + 
                     adv_weight * adv_loss_val)

        g_grads = tape.gradient(g_loss, encoder.trainable_weights + decoder.trainable_weights)
        if g_grads and all(g is not None for g in g_grads):
            generator_optimizer.apply_gradients(zip(g_grads, encoder.trainable_weights + decoder.trainable_weights))

        # Track losses
        epoch_losses['d_loss'] += d_loss.numpy()
        epoch_losses['g_loss'] += g_loss.numpy()
        epoch_losses['kl_loss'] += kl_loss_val.numpy()
        epoch_losses['recon_loss'] += recon_loss_val.numpy()

        if step % 50 == 0:
            if use_discriminator:
                print(f"Step {step}: D={d_loss.numpy():.4f}, G={g_loss.numpy():.4f}, "
                      f"KL={kl_loss_val.numpy():.4f}, Recon={recon_loss_val.numpy():.4f}")
            else:
                print(f"Step {step}: G={g_loss.numpy():.4f}, "
                      f"KL={kl_loss_val.numpy():.4f}, Recon={recon_loss_val.numpy():.4f}")

    # Epoch summary
    for key in epoch_losses:
        epoch_losses[key] /= steps_per_epoch
    
    print(f"Epoch {epoch + 1} Summary:")
    print(f"  D_loss: {epoch_losses['d_loss']:.4f}")
    print(f"  G_loss: {epoch_losses['g_loss']:.4f}")
    print(f"  KL_loss: {epoch_losses['kl_loss']:.4f}")
    print(f"  Recon_loss: {epoch_losses['recon_loss']:.4f}")

    # Early stopping with architecture-specific patience
    if epoch_losses['recon_loss'] < best_recon_loss:
        best_recon_loss = epoch_losses['recon_loss']
        wait = 0
        
        # Save models with architecture-specific naming
        arch_name = arch_info['name']
        encoder.save(f"{output_path}/{arch_name}_encoder_best.h5")
        decoder.save(f"{output_path}/{arch_name}_decoder_best.h5")
        if use_discriminator:
            discriminator.save(f"{output_path}/{arch_name}_discriminator_best.h5")
        print(f"✅ Models saved at epoch {epoch + 1} ({arch_name} architecture)")
    else:
        wait += 1
        if wait >= patience:
            print(f"⏹️ Early stopping at epoch {epoch + 1} (patience: {patience})")
            break

print(f"\n🎉 TRAINING COMPLETED!")
print(f"Best reconstruction loss: {best_recon_loss:.6f}")
print(f"Architecture used: {arch_info['name'].upper()}")

# === EVALUATION ===
print(f"\n📊 EVALUATING MODEL PERFORMANCE...")
scores, predictions, threshold = compute_anomaly_scores(encoder, decoder, X_test, X_train_normal)
print(f"Anomaly threshold: {threshold:.4f}")
print(f"Detected anomalies: {np.sum(predictions)}/{len(predictions)} ({np.sum(predictions)/len(predictions)*100:.1f}%)")

if y_test is not None:
    evaluate_with_labels(encoder, decoder, X_test, y_test)
    
    # Quick separation analysis
    normal_test_errors = scores[y_test == 0]
    fdia_test_errors = scores[y_test == 1]
    separation_ratio = np.mean(fdia_test_errors) / np.mean(normal_test_errors)
    
    print(f"\n🎯 SEPARATION ANALYSIS:")
    print(f"Normal test errors: {np.mean(normal_test_errors):.6f} ± {np.std(normal_test_errors):.6f}")
    print(f"FDIA test errors:   {np.mean(fdia_test_errors):.6f} ± {np.std(fdia_test_errors):.6f}")
    print(f"Separation ratio:   {separation_ratio:.3f}x")
    
    if separation_ratio > 2.0:
        print("✅ GOOD: Strong separation achieved!")
    elif separation_ratio > 1.5:
        print("⚠️ MODERATE: Some separation, may need tuning")
    else:
        print("❌ POOR: Weak separation, model needs improvement")

print(f"\n📁 Models saved to: {output_path}/{arch_info['name']}_*_best.h5")
print("✅ Training pipeline completed successfully!")