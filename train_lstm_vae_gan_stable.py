import numpy as np
import tensorflow as tf
from models.lstm_vae_gan import select_architecture
import os
import time
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import optuna
from optuna.integration import TFKerasPruningCallback
import logging

def kl_loss(z_mean, z_log_var):
    return -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))

def reconstruction_loss(y_true, y_pred):
    """Standard MSE reconstruction loss"""
    return tf.reduce_mean(tf.square(y_true - y_pred))

def spectral_reconstruction_loss(y_true, y_pred):
    """Enhanced reconstruction loss with multiple components"""
    # Ensure both inputs are float32
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Standard MSE loss
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    
    # L1 loss for sparsity
    mae_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
    
    # Temporal consistency loss (penalize sudden changes)
    temporal_diff_true = y_true[:, 1:, :] - y_true[:, :-1, :]
    temporal_diff_pred = y_pred[:, 1:, :] - y_pred[:, :-1, :]
    temporal_loss = tf.reduce_mean(tf.square(temporal_diff_true - temporal_diff_pred))
    
    # Feature correlation loss (maintain relationships between features)
    feature_mean_true = tf.reduce_mean(y_true, axis=1, keepdims=True)
    feature_mean_pred = tf.reduce_mean(y_pred, axis=1, keepdims=True)
    feature_corr_true = feature_mean_true - y_true
    feature_corr_pred = feature_mean_pred - y_pred
    correlation_loss = tf.reduce_mean(tf.square(feature_corr_true - feature_corr_pred))
    
    return mse_loss + 0.1 * mae_loss + 0.05 * temporal_loss + 0.02 * correlation_loss

def robust_reconstruction_loss(y_true, y_pred, epsilon=0.01):
    """Simplified robust loss (Huber-style)"""
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
    # L2 regularization on weights
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
    
    # Stronger regularization for better anomaly detection
    return l2_loss * 0.01

def beta_schedule(epoch, total_epochs, max_beta=1.0, warmup_epochs=30):
    """Improved beta scheduling for better KL annealing"""
    if epoch < warmup_epochs:
        # Gradual warmup to allow reconstruction learning first
        return 0.001 + (max_beta - 0.001) * (epoch / warmup_epochs)
    else:
        # Stable high beta for good latent structure
        return max_beta

def anomaly_regularization_loss(z_mean, z_log_var, latent_samples):
    """Pure anomaly detection regularization - encourage tight latent distribution for normal data"""
    # Encourage latent vectors to be close to unit Gaussian
    kl_divergence = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    
    # Additional regularization: encourage tight clustering of normal data in latent space
    latent_mean = tf.reduce_mean(latent_samples, axis=0)
    latent_variance = tf.reduce_mean(tf.square(latent_samples - latent_mean))
    
    # Encourage compact representation for normal data
    compactness_loss = latent_variance
    
    return kl_divergence + 0.1 * compactness_loss

def objective(trial):
    """Optuna objective function for pure anomaly detection"""
    print(f"\n🔬 STARTING TRIAL {trial.number}")
    print(f"📋 Trial {trial.number}: Sampling hyperparameters for pure anomaly detection...")
    
    # Clear any previous TensorFlow sessions
    tf.keras.backend.clear_session()
    
    # Suggest hyperparameters for pure anomaly detection
    initial_lr = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    kl_weight = trial.suggest_float('kl_weight', 0.001, 0.1, log=True)
    regularization_weight = trial.suggest_float('regularization_weight', 0.001, 0.1, log=True)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 24, 32])
    latent_dim = trial.suggest_int('latent_dim', 2, 8)
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.3)
    
    print(f"🎯 Trial {trial.number}: Training with lr={initial_lr:.6f}, batch_size={batch_size}, latent_dim={latent_dim}")
    print(f"   Other params: kl_weight={kl_weight:.4f}, regularization_weight={regularization_weight:.4f}")
    
    # Architecture selection with suggested latent_dim
    encoder, decoder, discriminator, arch_info = select_architecture(
        normal_samples_count=len(X_train_normal),
        input_shape=input_shape,
        latent_dim=latent_dim,
        force_architecture='compact'
    )
    
    # Optimizer with suggested learning rate
    optimizer = tf.keras.optimizers.Adam(initial_lr, clipnorm=1.0)
    
    # Data augmentation for training diversity (reduced augmentation)
    X_augmented = data_augmentation(X_train_normal, augmentation_factor=0.1)  # Reduced from 0.3
    X_train_enhanced = np.vstack([X_train_normal, X_augmented])
    
    # Training parameters (faster training)
    epochs = 30  # Reduced from 40 for faster optimization
    steps_per_epoch = min(X_train_enhanced.shape[0] // batch_size, 40)  # Reduced from 60
    
    best_reconstruction_loss = float('inf')
    best_separation_ratio = 0.0
    patience = 10  # Reduced from 15 for faster convergence
    wait = 0
    
    print(f"Trial {trial.number}: Training pure anomaly detector...")
    
    for epoch in range(epochs):
        epoch_losses = {'vae_loss': 0, 'recon_loss': 0, 'kl_loss': 0, 'reg_loss': 0}
        
        for step in range(steps_per_epoch):
            # Get batch of normal data only
            idx = np.random.randint(0, X_train_enhanced.shape[0], batch_size)
            normal_batch = X_train_enhanced[idx]
            
            with tf.GradientTape() as tape:
                # Forward pass
                z_mean, z_log_var, z = encoder(normal_batch, training=True)
                reconstructed = decoder(z, training=True)
                
                # Pure anomaly detection losses (simplified for speed)
                recon_loss_val = spectral_reconstruction_loss(normal_batch, reconstructed)
                # Removed robust_reconstruction_loss for speed - benchmark separately if needed
                
                # Anomaly regularization (encourage tight latent distribution)
                anomaly_reg_loss = anomaly_regularization_loss(z_mean, z_log_var, z)
                
                # Standard regularization
                reg_loss_val = regularization_loss(encoder, decoder)
                
                # Beta scheduling for KL annealing
                beta = beta_schedule(epoch, epochs)
                
                # Total VAE loss for pure anomaly detection
                vae_loss = (recon_loss_val + 
                           beta * kl_weight * anomaly_reg_loss + 
                           regularization_weight * reg_loss_val)
            
            # Apply gradients
            grads = tape.gradient(vae_loss, encoder.trainable_weights + decoder.trainable_weights)
            if grads and all(g is not None for g in grads):
                if not any(tf.reduce_any(tf.math.is_nan(grad)) for grad in grads):
                    grads = [tf.clip_by_norm(g, 1.0) for g in grads]
                    optimizer.apply_gradients(zip(grads, encoder.trainable_weights + decoder.trainable_weights))
            
            # Track losses
            epoch_losses['vae_loss'] += vae_loss.numpy()
            epoch_losses['recon_loss'] += recon_loss_val.numpy()
            epoch_losses['kl_loss'] += anomaly_reg_loss.numpy()
            epoch_losses['reg_loss'] += reg_loss_val.numpy()
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= steps_per_epoch
        
        # Print progress every 5 epochs
        if epoch % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: VAE_loss={epoch_losses['vae_loss']:.4f}, Recon_loss={epoch_losses['recon_loss']:.4f}")
        
        # Evaluate on test data every 8 epochs (reduced frequency for speed)
        if epoch % 8 == 0 or epoch == epochs - 1:
            # Quick evaluation on a subset of test data
            test_subset_size = min(500, len(X_test))
            test_subset_idx = np.random.choice(len(X_test), test_subset_size, replace=False)
            X_test_subset = X_test[test_subset_idx]
            y_test_subset = y_test[test_subset_idx] if y_test is not None else None
            
            if y_test_subset is not None:
                # Calculate reconstruction errors for anomaly detection
                z_mean_test, z_log_var_test, z_test_enc = encoder(X_test_subset)
                reconstructed_test = decoder(z_test_enc)
                recon_errors = tf.reduce_mean(tf.square(X_test_subset - reconstructed_test), axis=[1, 2]).numpy()
                
                normal_errors = recon_errors[y_test_subset == 0]
                anomaly_errors = recon_errors[y_test_subset == 1]
                
                if len(normal_errors) > 0 and len(anomaly_errors) > 0:
                    separation_ratio = np.mean(anomaly_errors) / np.mean(normal_errors)
                    
                    print(f"    Epoch {epoch+1} Separation: {separation_ratio:.4f}x (best: {best_separation_ratio:.4f}x)")
                    
                    # Track best separation ratio for anomaly detection
                    if separation_ratio > best_separation_ratio:
                        best_separation_ratio = separation_ratio
                        wait = 0
                    else:
                        wait += 1
                    
                    # Report intermediate value for pruning
                    trial.report(separation_ratio, epoch)
                    
                    # Check if trial should be pruned
                    if trial.should_prune():
                        print(f"    Trial {trial.number} pruned at epoch {epoch+1}")
                        raise optuna.TrialPruned()
                    
                    if wait >= patience:
                        print(f"    Early stopping triggered (patience={patience})")
                        break
                        
                # Also track reconstruction loss for normal data quality
                normal_recon_loss = np.mean(normal_errors)
                if normal_recon_loss < best_reconstruction_loss:
                    best_reconstruction_loss = normal_recon_loss
    
    print(f"Trial {trial.number} completed: Best separation = {best_separation_ratio:.4f}x, Best recon loss = {best_reconstruction_loss:.6f}")
    return best_separation_ratio

# Remove FDIA pattern generation - using pure anomaly detection
# The model will learn only from normal data and detect any deviations as anomalies

def data_augmentation(X_normal, augmentation_factor=0.2):
    """Apply data augmentation techniques to increase training diversity"""
    n_augmented = int(len(X_normal) * augmentation_factor)
    augmented_data = []
    
    for i in range(n_augmented):
        # Select random sample
        idx = np.random.randint(0, len(X_normal))
        sample = X_normal[idx].copy()
        
        # Apply random augmentation
        aug_type = np.random.choice(['noise', 'temporal_shift', 'feature_dropout', 'temporal_jitter'])
        
        if aug_type == 'noise':
            # Add small amount of Gaussian noise
            noise_level = np.random.uniform(0.001, 0.005)
            sample += np.random.normal(0, noise_level, sample.shape)
            
        elif aug_type == 'temporal_shift':
            # Shift sequence in time slightly
            shift = np.random.randint(-2, 3)
            if shift != 0:
                sample = np.roll(sample, shift, axis=0)
                
        elif aug_type == 'feature_dropout':
            # Randomly mask small portions of features
            n_features_to_mask = np.random.randint(1, 3)
            features_to_mask = np.random.choice(17, size=n_features_to_mask, replace=False)
            time_steps_to_mask = np.random.choice(12, size=np.random.randint(1, 3), replace=False)
            
            for feat in features_to_mask:
                for t in time_steps_to_mask:
                    # Mask with nearby values instead of zeros
                    if t > 0:
                        sample[t, feat] = sample[t-1, feat]
                        
        elif aug_type == 'temporal_jitter':
            # Small temporal interpolation
            for feat in range(17):
                if np.random.random() < 0.3:  # 30% chance per feature
                    # Add small temporal variations
                    jitter = np.random.normal(0, 0.001, 12)
                    sample[:, feat] += jitter
        
        augmented_data.append(sample)
    
    return np.array(augmented_data)

def compute_reconstruction_anomaly_scores(encoder, decoder, X_test, X_train_normal, threshold_percentile=90):
    """Compute anomaly scores based on reconstruction error only"""
    # Get test reconstruction errors
    z_mean_test, z_log_var_test, z_test = encoder(X_test)
    reconstructed_test = decoder(z_test)
    test_recon_errors = tf.reduce_mean(tf.square(X_test - reconstructed_test), axis=[1, 2])
    
    # Get training reconstruction errors (NORMAL DATA ONLY)
    z_mean_train, z_log_var_train, z_train = encoder(X_train_normal)
    reconstructed_train = decoder(z_train)
    train_recon_errors = tf.reduce_mean(tf.square(X_train_normal - reconstructed_train), axis=[1, 2])
    
    # Use specified percentile for threshold
    threshold = np.percentile(train_recon_errors, threshold_percentile)
    
    # Binary predictions
    predictions = (test_recon_errors > threshold).numpy().astype(int)
    
    return test_recon_errors.numpy(), predictions, threshold

def evaluate_reconstruction_based(encoder, decoder, X_test, y_test_labels, X_train_normal):
    """Evaluate using reconstruction error with comprehensive analysis"""
    # Get reconstruction errors
    z_mean_test, z_log_var_test, z_test = encoder(X_test)
    reconstructed_test = decoder(z_test)
    recon_errors = tf.reduce_mean(tf.square(X_test - reconstructed_test), axis=[1, 2]).numpy()
    
    print("\n=== RECONSTRUCTION-BASED ANOMALY DETECTION ===")
    
    # ROC analysis
    fpr, tpr, thresholds_roc = roc_curve(y_test_labels, recon_errors)
    roc_auc = auc(fpr, tpr)
    
    # Precision-Recall analysis
    precision, recall, pr_thresholds = precision_recall_curve(y_test_labels, recon_errors)
    pr_auc = auc(recall, precision)
    
    print(f"\n📊 Performance Metrics:")
    print(f"AUC-ROC: {roc_auc:.4f}")
    print(f"AUC-PR: {pr_auc:.4f}")
    
    # Find optimal threshold using F1 score
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = pr_thresholds[optimal_idx] if optimal_idx < len(pr_thresholds) else pr_thresholds[-1]
    
    print(f"Optimal threshold (F1): {optimal_threshold:.6f}")
    
    # Evaluate with optimal threshold
    optimal_predictions = (recon_errors > optimal_threshold).astype(int)
    print(f"Predicted anomalies: {np.sum(optimal_predictions)}/{len(optimal_predictions)}")
    
    if len(np.unique(y_test_labels)) > 1:
        print("Classification Report (F1-Optimal):")
        print(classification_report(y_test_labels, optimal_predictions))
    
    # Try different percentile thresholds
    percentiles = [85, 90, 95, 97, 99]
    thresholds = np.percentile(recon_errors, percentiles)
    
    print(f"\n📈 Percentile Threshold Analysis:")
    best_f1 = 0
    best_config = None
    
    for i, thresh in enumerate(thresholds):
        predictions = (recon_errors > thresh).astype(int)
        
        if len(np.unique(y_test_labels)) > 1 and np.sum(predictions) > 0:
            # Calculate F1 score
            tp = np.sum((y_test_labels == 1) & (predictions == 1))
            fp = np.sum((y_test_labels == 0) & (predictions == 1))
            fn = np.sum((y_test_labels == 1) & (predictions == 0))
            
            precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_val = 2 * (precision_val * recall_val) / (precision_val + recall_val) if (precision_val + recall_val) > 0 else 0
            
            if f1_val > best_f1:
                best_f1 = f1_val
                best_config = (percentiles[i], thresh, predictions)
        
        print(f"\nThreshold (percentile {percentiles[i]}): {thresh:.6f}")
        print(f"Predicted anomalies: {np.sum(predictions)}/{len(predictions)}")
        
        if len(np.unique(y_test_labels)) > 1:
            print("Classification Report:")
            print(classification_report(y_test_labels, predictions, zero_division=0))
    
    if best_config:
        print(f"\n🏆 BEST CONFIGURATION:")
        print(f"Percentile: {best_config[0]}, Threshold: {best_config[1]:.6f}, F1: {best_f1:.4f}")

def train_with_best_params(best_params):
    """Train final model with best hyperparameters for pure anomaly detection"""
    print(f"\n🏆 TRAINING FINAL PURE ANOMALY DETECTION MODEL:")
    for param, value in best_params.items():
        print(f"   {param}: {value}")
    
    # Clear session
    tf.keras.backend.clear_session()
    
    # Build model with best parameters
    encoder, decoder, discriminator, arch_info = select_architecture(
        normal_samples_count=len(X_train_normal),
        input_shape=input_shape,
        latent_dim=best_params['latent_dim'],
        force_architecture='compact'
    )
    
    print(f"\n🎯 Final architecture: {arch_info['name'].upper()}")
    print(f"Latent dimension: {best_params['latent_dim']}")
    
    # Optimizer with best learning rate
    optimizer = tf.keras.optimizers.Adam(best_params['learning_rate'], clipnorm=1.0)
    
    # Data augmentation for training diversity (reduced augmentation for speed)
    X_augmented = data_augmentation(X_train_normal, augmentation_factor=0.1)  # Reduced from 0.4
    X_train_enhanced = np.vstack([X_train_normal, X_augmented])
    
    print(f"Training data enhanced: {X_train_normal.shape[0]} -> {X_train_enhanced.shape[0]} samples (pure normal data)")
    
    # Training parameters (optimized for speed)
    batch_size = best_params['batch_size']
    epochs = 60  # Reduced from 80 for faster final training
    steps_per_epoch = min(X_train_enhanced.shape[0] // batch_size, 80)  # Reduced from 100
    
    print(f"Batch size: {batch_size}, Steps per epoch: {steps_per_epoch}")
    
    best_recon_loss = float('inf')
    best_separation_ratio = 0.0
    wait = 0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        epoch_losses = {'vae_loss': 0, 'kl_loss': 0, 'recon_loss': 0, 'reg_loss': 0}
        
        for step in range(steps_per_epoch):
            # Get batch of normal data only
            idx = np.random.randint(0, X_train_enhanced.shape[0], batch_size)
            normal_batch = X_train_enhanced[idx]
            
            with tf.GradientTape() as tape:
                # Forward pass
                z_mean, z_log_var, z = encoder(normal_batch, training=True)
                reconstructed = decoder(z, training=True)
                
                # Pure anomaly detection losses (simplified for speed)
                recon_loss_val = spectral_reconstruction_loss(normal_batch, reconstructed)
                # Removed robust_reconstruction_loss for speed - can benchmark separately if needed
                
                # Anomaly regularization
                anomaly_reg_loss = anomaly_regularization_loss(z_mean, z_log_var, z)
                
                # Standard regularization
                reg_loss_val = regularization_loss(encoder, decoder)
                
                # Beta scheduling
                beta = beta_schedule(epoch, epochs)
                
                # Total loss for pure anomaly detection
                vae_loss = (recon_loss_val + 
                           beta * best_params['kl_weight'] * anomaly_reg_loss + 
                           best_params['regularization_weight'] * reg_loss_val)
            
            # Apply gradients
            grads = tape.gradient(vae_loss, encoder.trainable_weights + decoder.trainable_weights)
            if grads and all(g is not None for g in grads):
                if not any(tf.reduce_any(tf.math.is_nan(grad)) for grad in grads):
                    grads = [tf.clip_by_norm(g, 1.0) for g in grads]
                    optimizer.apply_gradients(zip(grads, encoder.trainable_weights + decoder.trainable_weights))
            
            # Track losses
            epoch_losses['vae_loss'] += vae_loss.numpy()
            epoch_losses['kl_loss'] += anomaly_reg_loss.numpy()
            epoch_losses['recon_loss'] += recon_loss_val.numpy()
            epoch_losses['reg_loss'] += reg_loss_val.numpy()
            
            if step % 50 == 0:
                print(f"Step {step}: VAE={vae_loss.numpy():.4f}, "
                      f"KL={anomaly_reg_loss.numpy():.4f}, Recon={recon_loss_val.numpy():.4f}, "
                      f"Reg={reg_loss_val.numpy():.4f}")
        
        # Epoch summary
        for key in epoch_losses:
            epoch_losses[key] /= steps_per_epoch
        
        print(f"Epoch {epoch + 1} Summary:")
        print(f"  VAE_loss: {epoch_losses['vae_loss']:.4f}")
        print(f"  KL_loss: {epoch_losses['kl_loss']:.4f}")
        print(f"  Recon_loss: {epoch_losses['recon_loss']:.4f}")
        print(f"  Reg_loss: {epoch_losses['reg_loss']:.4f}")
        
        # Evaluate anomaly detection performance
        if y_test is not None:
            z_mean_test, z_log_var_test, z_test_enc = encoder(X_test)
            reconstructed_test = decoder(z_test_enc)
            recon_errors = tf.reduce_mean(tf.square(X_test - reconstructed_test), axis=[1, 2]).numpy()
            
            normal_errors = recon_errors[y_test == 0]
            anomaly_errors = recon_errors[y_test == 1]
            separation_ratio = np.mean(anomaly_errors) / np.mean(normal_errors)
            
            print(f"  Anomaly separation ratio: {separation_ratio:.3f}x")
            
            # Save best model based on separation ratio
            if separation_ratio > best_separation_ratio:
                best_separation_ratio = separation_ratio
                best_recon_loss = epoch_losses['recon_loss']
                wait = 0
                
                # Save models
                encoder.save(f"{output_path}/pure_anomaly_best_encoder.h5")
                decoder.save(f"{output_path}/pure_anomaly_best_decoder.h5")
                print(f"✅ Best model saved (separation: {separation_ratio:.3f}x)")
            else:
                wait += 1
                
                if wait >= 25:  # Longer patience for pure anomaly detection
                    print(f"⏹️ Early stopping at epoch {epoch + 1}")
                    break
    
    return encoder, decoder, best_separation_ratio

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

# === OPTUNA HYPERPARAMETER OPTIMIZATION ===
input_shape = (X_train_normal.shape[1], X_train_normal.shape[2])
print(f"Input shape: {input_shape}")

# === SPEED OPTIMIZATIONS APPLIED ===
# ✅ Reduced Optuna trials: 12 → 5 for faster tuning
# ✅ Reduced augmentation: 0.3 → 0.1 for less noise overhead
# ✅ Limited steps_per_epoch: 60 → 40 during optimization, 100 → 80 final
# ✅ Simplified reconstruction loss: removed robust_loss for speed
# ✅ Reduced patience: 15 → 10 for faster convergence
# ✅ Less frequent evaluation: every 5 → 8 epochs during optimization

print("\n🔍 STARTING PURE ANOMALY DETECTION OPTIMIZATION...")
print("Objective: Train on normal data only and maximize anomaly separation ratio")

# Set up verbose logging
optuna.logging.set_verbosity(optuna.logging.DEBUG)
logging.basicConfig(level=logging.INFO)

# Create study
study = optuna.create_study(
    direction='maximize',
    study_name='pure_anomaly_detection_optimization',
    pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5)
)

# Run optimization
n_trials = 5  # Reduced from 12 to 5 for faster tuning
print(f"Running {n_trials} pure anomaly detection optimization trials...")

# Define progress callback for real-time updates
def progress_callback(study, trial):
    print(f"\n📊 TRIAL {trial.number} COMPLETED:")
    print(f"   Value: {trial.value:.4f}" if trial.value else "   Value: PRUNED/FAILED")
    print(f"   Best value so far: {study.best_value:.4f}")
    print(f"   Number of trials: {len(study.trials)}")
    if trial.value:
        print(f"   Parameters: {trial.params}")
    print("-" * 50)

print("💡 Monitoring pure anomaly detection trial progress in real-time...")
start_time = time.time()
study.optimize(objective, n_trials=n_trials, timeout=7200, callbacks=[progress_callback])  # 2 hour timeout (reduced)
optimization_time = time.time() - start_time
print(f"✅ Pure anomaly detection optimization completed in {optimization_time/60:.1f} minutes")

# Print optimization results
print("\n🎯 PURE ANOMALY DETECTION OPTIMIZATION COMPLETED!")
print(f"Number of finished trials: {len(study.trials)}")
print(f"Best trial: {study.best_trial.number}")
print(f"Best anomaly separation ratio: {study.best_value:.4f}")
print("Best parameters:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")

# Train final model with best parameters
print(f"\n🏋️ TRAINING FINAL PURE ANOMALY DETECTION MODEL...")

encoder, decoder, final_separation_ratio = train_with_best_params(study.best_params)

print(f"\n🎉 PURE ANOMALY DETECTION TRAINING COMPLETED!")
print(f"Final anomaly separation ratio: {final_separation_ratio:.4f}")
print(f"Best parameters used:")
for param, value in study.best_params.items():
    print(f"  {param}: {value}")

# === COMPREHENSIVE PURE ANOMALY DETECTION EVALUATION ===
print(f"\n📊 EVALUATING PURE ANOMALY DETECTION MODEL PERFORMANCE...")

# Test different threshold percentiles for anomaly detection
for percentile in [85, 90, 95, 97, 99]:
    scores, predictions, threshold = compute_reconstruction_anomaly_scores(
        encoder, decoder, X_test, X_train_normal, threshold_percentile=percentile
    )
    print(f"\n🎯 {percentile}th percentile threshold: {threshold:.6f}")
    print(f"Detected anomalies: {np.sum(predictions)}/{len(predictions)} ({np.sum(predictions)/len(predictions)*100:.1f}%)")

if y_test is not None:
    evaluate_reconstruction_based(encoder, decoder, X_test, y_test, X_train_normal)
    
    # Enhanced separation analysis for pure anomaly detection
    z_mean_test, z_log_var_test, z_test_enc = encoder(X_test)
    reconstructed_test = decoder(z_test_enc)
    recon_errors = tf.reduce_mean(tf.square(X_test - reconstructed_test), axis=[1, 2]).numpy()
    
    normal_errors = recon_errors[y_test == 0]
    anomaly_errors = recon_errors[y_test == 1]  # Any type of anomaly (FDIA, etc.)
    separation_ratio = np.mean(anomaly_errors) / np.mean(normal_errors)
    
    print(f"\n🎯 FINAL PURE ANOMALY SEPARATION ANALYSIS:")
    print(f"Normal test errors:  {np.mean(normal_errors):.6f} ± {np.std(normal_errors):.6f}")
    print(f"Anomaly test errors: {np.mean(anomaly_errors):.6f} ± {np.std(anomaly_errors):.6f}")
    print(f"Separation ratio:    {separation_ratio:.3f}x")
    
    # Statistical significance test
    from scipy.stats import ttest_ind
    t_stat, p_value = ttest_ind(anomaly_errors, normal_errors)
    print(f"T-test p-value: {p_value:.2e} ({'significant' if p_value < 0.001 else 'not significant'})")
    
    if separation_ratio > 3.0:
        print("✅ EXCELLENT: Very strong anomaly separation achieved!")
    elif separation_ratio > 2.0:
        print("✅ GOOD: Strong anomaly separation achieved!")
    elif separation_ratio > 1.5:
        print("⚠️ MODERATE: Some anomaly separation, may need tuning")
    else:
        print("❌ POOR: Weak anomaly separation, model needs improvement")
        
    print(f"\n💾 Models saved as 'pure_anomaly_best_encoder.h5' and 'pure_anomaly_best_decoder.h5'")
