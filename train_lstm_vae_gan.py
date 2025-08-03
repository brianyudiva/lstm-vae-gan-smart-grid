import numpy as np
import tensorflow as tf
from models.lstm_vae_gan import select_architecture
import os
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve

def kl_loss(z_mean, z_log_var):
    return -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))

def reconstruction_loss(y_true, y_pred):
    """Standard MSE reconstruction loss for regular model"""
    return tf.reduce_mean(tf.square(y_true - y_pred))

def contrastive_reconstruction_loss(y_true, y_pred, margin=0.005):
    """Enhanced reconstruction loss with adaptive margin"""
    mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=[1, 2])
    # Use smaller margin and adaptive penalty
    adaptive_margin = margin * tf.reduce_mean(mse)  # Adaptive to current reconstruction quality
    penalty = tf.maximum(0.0, mse - adaptive_margin) * 10.0  # Reduced penalty multiplier
    return tf.reduce_mean(mse + penalty)

def focal_reconstruction_loss(y_true, y_pred, alpha=2.0, gamma=2.0):
    """Focal loss variant for reconstruction to focus on hard examples"""
    mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=[1, 2])
    # Focus more on samples with higher reconstruction error
    focal_weight = tf.pow(mse / (tf.reduce_mean(mse) + 1e-8), gamma)
    return tf.reduce_mean(alpha * focal_weight * mse)

def compute_anomaly_scores(encoder, decoder, X_test, X_train_normal, threshold_percentile=95):
    """Compute anomaly scores with multiple metrics and ensemble approach"""
    # Get test reconstruction errors
    z_mean_test, z_log_var_test, z_test = encoder(X_test)
    reconstructed_test = decoder(z_test)
    test_recon_errors = tf.reduce_mean(tf.square(X_test - reconstructed_test), axis=[1, 2])
    
    # Get training reconstruction errors (NORMAL DATA ONLY)
    z_mean_train, z_log_var_train, z_train = encoder(X_train_normal)
    reconstructed_train = decoder(z_train)
    train_recon_errors = tf.reduce_mean(tf.square(X_train_normal - reconstructed_train), axis=[1, 2])
    
    # Additional anomaly metrics
    # 1. KL divergence from prior (latent space regularization)
    kl_div_test = 0.5 * tf.reduce_sum(tf.square(z_mean_test) + tf.exp(z_log_var_test) - z_log_var_test - 1, axis=1)
    kl_div_train = 0.5 * tf.reduce_sum(tf.square(z_mean_train) + tf.exp(z_log_var_train) - z_log_var_train - 1, axis=1)
    
    # 2. Latent space distance from training distribution
    train_latent_mean = tf.reduce_mean(z_train, axis=0)
    latent_distance_test = tf.reduce_mean(tf.square(z_test - train_latent_mean), axis=1)
    latent_distance_train = tf.reduce_mean(tf.square(z_train - train_latent_mean), axis=1)
    
    # Ensemble scoring with weighted combination
    recon_weight = 0.7  # Primary metric
    kl_weight = 0.2     # Secondary metric
    latent_weight = 0.1 # Tertiary metric
    
    # Normalize scores to [0,1] range
    def normalize_scores(scores):
        return (scores - tf.reduce_min(scores)) / (tf.reduce_max(scores) - tf.reduce_min(scores) + 1e-8)
    
    test_recon_norm = normalize_scores(test_recon_errors)
    test_kl_norm = normalize_scores(kl_div_test)
    test_latent_norm = normalize_scores(latent_distance_test)
    
    train_recon_norm = normalize_scores(train_recon_errors)
    train_kl_norm = normalize_scores(kl_div_train)
    train_latent_norm = normalize_scores(latent_distance_train)
    
    # Combined anomaly scores
    test_combined_scores = (recon_weight * test_recon_norm + 
                           kl_weight * test_kl_norm + 
                           latent_weight * test_latent_norm)
    
    train_combined_scores = (recon_weight * train_recon_norm + 
                            kl_weight * train_kl_norm + 
                            latent_weight * train_latent_norm)
    
    # Set threshold based on TRAINING data with combined scores
    threshold = np.percentile(train_combined_scores, threshold_percentile)
    
    # Binary predictions using combined scores
    predictions = (test_combined_scores > threshold).numpy().astype(int)
    
    return test_combined_scores.numpy(), predictions, threshold

def evaluate_with_labels(encoder, decoder, X_test, y_test_labels, X_train_normal):
    """Enhanced evaluation with multiple metrics and optimal threshold finding"""
    # Get combined anomaly scores
    combined_scores, _, _ = compute_anomaly_scores(encoder, decoder, X_test, X_train_normal, threshold_percentile=95)
    
    # Also get traditional reconstruction errors for comparison
    z_mean_test, z_log_var_test, z_test = encoder(X_test)
    reconstructed_test = decoder(z_test)
    recon_errors = tf.reduce_mean(tf.square(X_test - reconstructed_test), axis=[1, 2]).numpy()
    
    print("\n=== ENHANCED EVALUATION WITH ACTUAL LABELS ===")
    
    # Find optimal threshold using ROC curve
    # ROC analysis for combined scores
    fpr, tpr, thresholds_roc = roc_curve(y_test_labels, combined_scores)
    roc_auc = auc(fpr, tpr)
    
    # Find optimal threshold (Youden's J statistic)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds_roc[optimal_idx]
    
    print(f"\n📊 ROC Analysis (Combined Scores):")
    print(f"AUC-ROC: {roc_auc:.4f}")
    print(f"Optimal threshold: {optimal_threshold:.6f}")
    
    # Evaluate with optimal threshold
    optimal_predictions = (combined_scores > optimal_threshold).astype(int)
    print(f"\nOptimal Threshold Results:")
    print(f"Predicted anomalies: {np.sum(optimal_predictions)}/{len(optimal_predictions)}")
    
    if len(np.unique(y_test_labels)) > 1:
        print("Classification Report (Optimal Threshold):")
        print(classification_report(y_test_labels, optimal_predictions))
    
    # Compare with traditional reconstruction-only approach
    print(f"\n📊 Comparison: Combined vs Reconstruction-only:")
    fpr_recon, tpr_recon, _ = roc_curve(y_test_labels, recon_errors)
    roc_auc_recon = auc(fpr_recon, tpr_recon)
    print(f"Combined approach AUC: {roc_auc:.4f}")
    print(f"Reconstruction-only AUC: {roc_auc_recon:.4f}")
    print(f"Improvement: {((roc_auc - roc_auc_recon) / roc_auc_recon * 100):+.2f}%")
    
    # Try different percentile thresholds for combined scores
    thresholds = np.percentile(combined_scores, [90, 95, 97, 99])
    
    print(f"\n📈 Percentile Threshold Analysis:")
    for i, thresh in enumerate(thresholds):
        predictions = (combined_scores > thresh).astype(int)
        print(f"\nThreshold (percentile {[90, 95, 97, 99][i]}): {thresh:.6f}")
        print(f"Predicted anomalies: {np.sum(predictions)}/{len(predictions)}")
        
        if len(np.unique(y_test_labels)) > 1:
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

# Force ultra-compact architecture for better generalization
encoder, decoder, discriminator, arch_info = select_architecture(
    normal_samples_count=len(X_train_normal),
    input_shape=input_shape,
    latent_dim=4,  # Reduced from 8
    force_architecture='ultra_compact'  # Force the simplest model
)

print(f"\n🎯 Selected architecture: {arch_info['name'].upper()}")

# Check if discriminator exists
use_discriminator = discriminator is not None
print(f"Using discriminator: {use_discriminator}")

# === COMPILE MODELS WITH ULTRA-AGGRESSIVE HYPERPARAMETERS ===
# Improved hyperparameters for better anomaly detection
initial_lr = 0.001   # Higher learning rate for faster convergence
kl_weight = 0.0001   # Much lower KL weight - focus on reconstruction
recon_weight = 1000  # Strong reconstruction weight but not excessive
adv_weight = 0.1     # Small adversarial weight for regularization
beta_schedule = True # Use beta-VAE scheduling for better disentanglement

print(f"\n🎛️ IMPROVED TRAINING CONFIGURATION:")
print(f"Learning rate: {initial_lr}")
print(f"Loss weights - KL: {kl_weight}, Recon: {recon_weight}, Adv: {adv_weight}")
print(f"🎯 FOCUS: Balanced training with beta-VAE scheduling")
print(f"Beta scheduling: {beta_schedule}")

generator_optimizer = tf.keras.optimizers.Adam(initial_lr)
# Add learning rate decay for better convergence
lr_decay_factor = 0.95
lr_decay_patience = 10
if use_discriminator:
    discriminator_optimizer = tf.keras.optimizers.Adam(initial_lr)
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)

# === TRAINING PARAMETERS ===
batch_size = 32      # Increased batch size for more stable gradients
epochs = 200         # Reasonable number of epochs
steps_per_epoch = X_train_normal.shape[0] // batch_size

# Beta-VAE scheduling parameters
beta_start = 0.0001  # Start with very low KL weight
beta_end = kl_weight # End with target KL weight
beta_warmup_epochs = 50  # Gradual increase over first 50 epochs

# Adjust training parameters based on architecture
if arch_info['name'] == 'ultra_compact':
    epochs = 300     # Much more epochs for ultra-compact
    patience = 50    # More patience for convergence
elif arch_info['name'] == 'compact':
    epochs = 200
    patience = 35
else:
    epochs = 150
    patience = 25

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
print(f"\n🏋️ STARTING ENHANCED TRAINING...")
print(f"Beta-VAE scheduling: {beta_start:.6f} → {beta_end:.6f} over {beta_warmup_epochs} epochs")
best_recon_loss = float('inf')
wait = 0
lr_wait = 0

for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    epoch_losses = {'d_loss': 0, 'g_loss': 0, 'kl_loss': 0, 'recon_loss': 0}
    
    # Beta-VAE scheduling - gradually increase KL weight
    if beta_schedule and epoch < beta_warmup_epochs:
        current_kl_weight = beta_start + (beta_end - beta_start) * (epoch / beta_warmup_epochs)
    else:
        current_kl_weight = kl_weight
    
    if epoch % 10 == 0:  # Print beta schedule updates
        print(f"Current KL weight: {current_kl_weight:.6f}")

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
            # Use focal reconstruction loss for better hard example focus
            recon_loss_val = focal_reconstruction_loss(real_seq, reconstructed)
            
            if use_discriminator:
                fake_pred = discriminator(reconstructed)
                adv_loss_val = bce(tf.ones_like(fake_pred), fake_pred)
            else:
                adv_loss_val = tf.constant(0.0)
            
            # Use scheduled KL weight
            g_loss = (current_kl_weight * kl_loss_val + 
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
        lr_wait = 0
        
        # Save models with architecture-specific naming
        arch_name = arch_info['name']
        encoder.save(f"{output_path}/{arch_name}_encoder_best.h5")
        decoder.save(f"{output_path}/{arch_name}_decoder_best.h5")
        if use_discriminator:
            discriminator.save(f"{output_path}/{arch_name}_discriminator_best.h5")
        print(f"✅ Models saved at epoch {epoch + 1} ({arch_name} architecture)")
    else:
        wait += 1
        lr_wait += 1
        
        # Learning rate decay
        if lr_wait >= lr_decay_patience:
            old_lr = generator_optimizer.learning_rate.numpy()
            new_lr = old_lr * lr_decay_factor
            generator_optimizer.learning_rate.assign(new_lr)
            lr_wait = 0
            print(f"🔽 Learning rate decayed: {old_lr:.6f} → {new_lr:.6f}")
        
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
    evaluate_with_labels(encoder, decoder, X_test, y_test, X_train_normal)
    
    # Enhanced separation analysis with combined scores
    combined_scores, _, _ = compute_anomaly_scores(encoder, decoder, X_test, X_train_normal)
    normal_test_scores = combined_scores[y_test == 0]
    fdia_test_scores = combined_scores[y_test == 1]
    separation_ratio = np.mean(fdia_test_scores) / np.mean(normal_test_scores)
    
    print(f"\n🎯 ENHANCED SEPARATION ANALYSIS:")
    print(f"Normal test scores: {np.mean(normal_test_scores):.6f} ± {np.std(normal_test_scores):.6f}")
    print(f"FDIA test scores:   {np.mean(fdia_test_scores):.6f} ± {np.std(fdia_test_scores):.6f}")
    print(f"Separation ratio:   {separation_ratio:.3f}x")
    
    # Also show traditional reconstruction-based separation for comparison
    z_mean_test, z_log_var_test, z_test = encoder(X_test)
    reconstructed_test = decoder(z_test)
    recon_errors = tf.reduce_mean(tf.square(X_test - reconstructed_test), axis=[1, 2]).numpy()
    
    normal_recon_errors = recon_errors[y_test == 0]
    fdia_recon_errors = recon_errors[y_test == 1]
    recon_separation_ratio = np.mean(fdia_recon_errors) / np.mean(normal_recon_errors)
    
    print(f"\n📊 Comparison - Combined vs Reconstruction-only:")
    print(f"Combined separation:      {separation_ratio:.3f}x")
    print(f"Reconstruction-only:      {recon_separation_ratio:.3f}x")
    print(f"Improvement factor:       {(separation_ratio / recon_separation_ratio):.3f}x")
    
    if separation_ratio > 3.0:
        print("✅ EXCELLENT: Very strong separation achieved!")
    elif separation_ratio > 2.0:
        print("✅ GOOD: Strong separation achieved!")
    elif separation_ratio > 1.5:
        print("⚠️ MODERATE: Some separation, may need tuning")
    else:
        print("❌ POOR: Weak separation, model needs improvement")

print(f"\n📁 Models saved to: {output_path}/{arch_info['name']}_*_best.h5")
print("✅ Training pipeline completed successfully!")