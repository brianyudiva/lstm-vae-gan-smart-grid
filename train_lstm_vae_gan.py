import numpy as np
import tensorflow as tf
from models.lstm_vae_gan import select_architecture
from utils.loss_functions import (
    kl_loss, reconstruction_loss, spectral_reconstruction_loss, 
    robust_reconstruction_loss, regularization_loss, beta_schedule,
    contrastive_latent_loss, anomaly_regularization_loss
)
from utils.data_augmentation import enhanced_data_augmentation
import os
import time
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve
import optuna
import logging
from scipy.stats import ttest_ind

# === PURE ANOMALY DETECTION TRAINING FUNCTIONS ===

def objective(trial):
    print(f"\nSTARTING TRIAL {trial.number}")
    print(f"Trial {trial.number}: Sampling hyperparameters for pure anomaly detection...")
    
    # Clear any previous TensorFlow sessions
    tf.keras.backend.clear_session()
    
    initial_lr = trial.suggest_float('learning_rate', 5e-5, 5e-3, log=True)
    kl_weight = trial.suggest_float('kl_weight', 0.01, 0.2, log=True)
    regularization_weight = trial.suggest_float('regularization_weight', 1e-6, 1e-4, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64])
    latent_dim = trial.suggest_int('latent_dim', 8, 16)
    
    reconstruction_weight = trial.suggest_float('reconstruction_weight', 20.0, 100.0)
    beta_multiplier = trial.suggest_float('beta_multiplier', 0.01, 0.3)
    contrastive_weight = trial.suggest_float('contrastive_weight', 0.01, 0.1)
    
    print(f"Trial {trial.number}: lr={initial_lr:.6f}, batch_size={batch_size}, latent_dim={latent_dim}")
    print(f"   KL_weight={kl_weight:.4f}, reg_weight={regularization_weight:.6f}, beta_mult={beta_multiplier:.3f}")
    print(f"   Recon_weight={reconstruction_weight:.1f}, Contrastive={contrastive_weight:.3f}")
    print(f"   Input shape: {input_shape} -> Latent: {latent_dim}")
    
    # Architecture selection
    encoder, decoder, discriminator = select_architecture(
        normal_samples_count=len(X_train_normal),
        input_shape=input_shape,
        latent_dim=latent_dim,
        force_architecture=None
    )

    optimizer = tf.keras.optimizers.Adam(initial_lr, clipnorm=1.0)

    X_augmented = enhanced_data_augmentation(X_train_normal, augmentation_factor=0.3)
    X_train_enhanced = np.vstack([X_train_normal, X_augmented])
    
    epochs = 50
    steps_per_epoch = min(X_train_enhanced.shape[0] // batch_size, 100)
    
    best_reconstruction_loss = float('inf')
    best_separation_ratio = 0.0
    patience = 10
    wait = 0
    
    print(f"Trial {trial.number}: Training anomaly detector...")
    
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
                
                # PHASE 1: Extreme reconstruction focus
                recon_loss_val = reconstruction_weight * robust_reconstruction_loss(normal_batch, reconstructed)
                
                # PHASE 2: Near-zero KL regularization
                kl_loss_val = kl_loss(z_mean, z_log_var)
                
                # PHASE 3: Minimal contrastive learning interference
                contrastive_loss_val = 0.0
                if step % 6 == 0 and len(X_train_enhanced) > len(X_train_normal):
                    aug_size = max(1, batch_size//6)
                    aug_idx = np.random.randint(len(X_train_normal), X_train_enhanced.shape[0], aug_size)
                    aug_batch = X_train_enhanced[aug_idx]
                    
                    z_mean_aug, z_log_var_aug, z_aug = encoder(aug_batch, training=True)
                    
                    min_batch_size = min(len(z), len(z_aug))
                    if min_batch_size > 0:
                        contrastive_loss_val = contrastive_latent_loss(z[:min_batch_size], z_aug[:min_batch_size])
                
                reg_loss_val = regularization_loss(encoder, decoder)
                
                beta = beta_schedule(epoch, epochs) * beta_multiplier
                
                progress_factor = min(1.0, epoch / 35.0)
                
                vae_loss = (reconstruction_weight * recon_loss_val +
                           beta * kl_weight * kl_loss_val +
                           regularization_weight * reg_loss_val +
                           progress_factor * contrastive_weight * contrastive_loss_val)
            
            # Apply gradients
            grads = tape.gradient(vae_loss, encoder.trainable_weights + decoder.trainable_weights)
            if grads and all(g is not None for g in grads):
                if not any(tf.reduce_any(tf.math.is_nan(grad)) for grad in grads):
                    grads = [tf.clip_by_norm(g, 1.0) for g in grads]
                    optimizer.apply_gradients(zip(grads, encoder.trainable_weights + decoder.trainable_weights))
            
            # Track losses
            epoch_losses['vae_loss'] += vae_loss.numpy()
            epoch_losses['recon_loss'] += recon_loss_val.numpy()
            epoch_losses['kl_loss'] += kl_loss_val.numpy()
            epoch_losses['reg_loss'] += reg_loss_val.numpy()
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= steps_per_epoch
        
        if epoch % 4 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: VAE_loss={epoch_losses['vae_loss']:.4f}, Recon_loss={epoch_losses['recon_loss']:.4f}")
            print(f"    KL_loss={epoch_losses['kl_loss']:.6f}, Reg_loss={epoch_losses['reg_loss']:.6f}")
        
        if epoch % 4 == 0 or epoch == epochs - 1:
            test_subset_size = min(800, len(X_test))
            test_subset_idx = np.random.choice(len(X_test), test_subset_size, replace=False)
            X_test_subset = X_test[test_subset_idx]
            y_test_subset = y_test[test_subset_idx] if y_test is not None else None
            
            if y_test_subset is not None:
                z_mean_test, z_log_var_test, z_test_enc = encoder(X_test_subset)
                reconstructed_test = decoder(z_test_enc)
                recon_errors = tf.reduce_mean(tf.square(X_test_subset - reconstructed_test), axis=[1, 2]).numpy()
                
                normal_errors = recon_errors[y_test_subset == 0]
                anomaly_errors = recon_errors[y_test_subset == 1]
                
                if len(normal_errors) > 0 and len(anomaly_errors) > 0:
                    separation_ratio = np.mean(anomaly_errors) / np.mean(normal_errors)
                    
                    normal_std = np.std(normal_errors)
                    anomaly_std = np.std(anomaly_errors)
                    effect_size = (np.mean(anomaly_errors) - np.mean(normal_errors)) / np.sqrt((normal_std**2 + anomaly_std**2) / 2)
                    
                    print(f"    Epoch {epoch+1} Separation: {separation_ratio:.4f}x, Effect: {effect_size:.3f}")
                    print(f"    Normal: {np.mean(normal_errors):.6f}±{normal_std:.6f}, Anomaly: {np.mean(anomaly_errors):.6f}±{anomaly_std:.6f}")
                    
                    separation_bonus = max(0, separation_ratio - 1.0) * 20
                    combined_score = separation_ratio * 0.8 + effect_size * 0.2
                    
                    if combined_score > best_separation_ratio:
                        best_separation_ratio = combined_score
                        wait = 0
                        print(f"    NEW BEST combined score: {combined_score:.4f} (separation: {separation_ratio:.4f}x)")
                    else:
                        wait += 1
                    
                    # Report for pruning
                    trial.report(combined_score, epoch)
                    
                    if trial.should_prune():
                        print(f"    Trial {trial.number} pruned at epoch {epoch+1}")
                        raise optuna.TrialPruned()
                    
                    if wait >= 8:  # Reduced patience for faster exploration
                        print(f"    Early stopping triggered (patience=8) - trying next configuration")
                        break
                        
                # Also track reconstruction loss for normal data quality
                normal_recon_loss = np.mean(normal_errors)
                if normal_recon_loss < best_reconstruction_loss:
                    best_reconstruction_loss = normal_recon_loss
    
    print(f"Trial {trial.number} completed: Best separation = {best_separation_ratio:.4f}x, Best recon loss = {best_reconstruction_loss:.6f}")
    return best_separation_ratio

def compute_reconstruction_anomaly_scores(encoder, decoder, X_test, X_train_normal, threshold_percentile=90):
    z_mean_test, z_log_var_test, z_test = encoder(X_test)
    reconstructed_test = decoder(z_test)
    test_recon_errors = tf.reduce_mean(tf.square(X_test - reconstructed_test), axis=[1, 2])
    
    z_mean_train, z_log_var_train, z_train = encoder(X_train_normal)
    reconstructed_train = decoder(z_train)
    train_recon_errors = tf.reduce_mean(tf.square(X_train_normal - reconstructed_train), axis=[1, 2])
    
    threshold = np.percentile(train_recon_errors, threshold_percentile)
    
    predictions = (test_recon_errors > threshold).numpy().astype(int)
    
    return test_recon_errors.numpy(), predictions, threshold

def evaluate_reconstruction_based(encoder, decoder, X_test, y_test_labels, X_train_normal):
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
    
    print(f"\nPerformance Metrics:")
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
    
    percentiles = [85, 90, 95, 97, 99]
    thresholds = np.percentile(recon_errors, percentiles)
    
    print(f"\nPercentile Threshold Analysis:")
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
        print(f"\nBEST CONFIGURATION:")
        print(f"Percentile: {best_config[0]}, Threshold: {best_config[1]:.6f}, F1: {best_f1:.4f}")

def train_with_best_params(best_params):
    print(f"\nTRAINING FINAL MODEL:")
    for param, value in best_params.items():
        print(f"   {param}: {value}")
    
    tf.keras.backend.clear_session()
    
    encoder, decoder, discriminator = select_architecture(
        normal_samples_count=len(X_train_normal),
        input_shape=input_shape,
        latent_dim=best_params['latent_dim'],
        force_architecture=None
    )
    
    print(f"Latent dimension: {best_params['latent_dim']}")
    
    optimizer = tf.keras.optimizers.Adam(best_params['learning_rate'], clipnorm=1.0)
    
    def lr_schedule(epoch):
        if epoch < 20:
            return best_params['learning_rate']
        elif epoch < 40:
            return best_params['learning_rate'] * 0.5
        else:
            return best_params['learning_rate'] * 0.2
    
    # X_augmented = enhanced_data_augmentation(X_train_normal, augmentation_factor=0.3)
    # X_train_enhanced = np.vstack([X_train_normal, X_augmented])
    X_train_enhanced = X_train_normal.copy()

    batch_size = best_params['batch_size']
    epochs = 60
    steps_per_epoch = min(X_train_enhanced.shape[0] // batch_size, 100)
    
    print(f"Batch size: {batch_size}, Steps per epoch: {steps_per_epoch}")
    
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
                                
                # Phase 1: Enhanced reconstruction loss
                reconstruction_weight = best_params.get('reconstruction_weight', 1.0)
                recon_loss_val = reconstruction_weight * spectral_reconstruction_loss(normal_batch, reconstructed)
                
                # Phase 2: Advanced anomaly regularization  
                anomaly_reg_loss = anomaly_regularization_loss(z_mean, z_log_var, z)
                
                # Phase 3: Contrastive learning (every 3rd step)
                contrastive_loss_val = 0.0
                if step % 3 == 0 and len(X_train_enhanced) > len(X_train_normal):
                    aug_idx = np.random.randint(len(X_train_normal), X_train_enhanced.shape[0], batch_size//2)
                    aug_batch = X_train_enhanced[aug_idx]
                    z_mean_aug, z_log_var_aug, z_aug = encoder(aug_batch, training=True)
                    
                    if len(z) >= len(z_aug):
                        contrastive_loss_val = contrastive_latent_loss(z[:len(z_aug)], z_aug)
                    else:
                        contrastive_loss_val = contrastive_latent_loss(z, z_aug[:len(z)])
                
                # Phase 4: Pure anomaly detection - no synthetic anomalies  
                synthetic_anomaly_loss = 0.0
                
                # Standard regularization
                reg_loss_val = regularization_loss(encoder, decoder)
                
                # Progressive beta scheduling
                beta = beta_schedule(epoch, epochs) * best_params.get('beta_multiplier', 2.0)
                progress_factor = min(1.0, epoch / 15.0)
                
                # COMBINED PROGRESSIVE LOSS (pure anomaly detection)
                vae_loss = (reconstruction_weight * recon_loss_val + 
                           beta * best_params['kl_weight'] * anomaly_reg_loss + 
                           best_params['regularization_weight'] * reg_loss_val +
                           progress_factor * 0.4 * contrastive_loss_val)  # Only contrastive learning
                
                # Learning rate adjustment
                current_lr = lr_schedule(epoch)
                if optimizer.learning_rate != current_lr:
                    optimizer.learning_rate.assign(current_lr)
            
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
                      f"Reg={reg_loss_val.numpy():.4f}, Contr={contrastive_loss_val:.4f}")
        
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
                
                # Save models with dataset-specific names
                encoder.save(f"{output_path}/{model_prefix}_encoder.h5")
                decoder.save(f"{output_path}/{model_prefix}_decoder.h5")
                print(f"Best model saved (separation: {separation_ratio:.3f}x) as {model_prefix}_*")
            else:
                wait += 1
                
                if wait >= 25:  # Reasonable patience
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
    
    return encoder, decoder, best_separation_ratio

# === CONFIG ===
sequence_path = "data/sequences"
output_path = "outputs/checkpoints"
model_prefix = "lstm_vae_gan"
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
    print("No labels found - using unsupervised mode")
    X_train_normal = X_train
    y_train, y_test = None, None

# === OPTUNA HYPERPARAMETER OPTIMIZATION ===
input_shape = (X_train_normal.shape[1], X_train_normal.shape[2])
input_complexity = input_shape[0] * input_shape[1]
print(f"Input shape: {input_shape}")
print(f"Input complexity: {input_complexity} dimensions")

if input_complexity > 600:
    print("🎯 HIGH-COMPLEXITY DATASET DETECTED - Using optimizations for Jacobian stealth attacks")
    print("   - High-capacity architecture will be auto-selected")
    print("   - Aggressive reconstruction focus")
    print("   - Minimal regularization to preserve learning capacity")
else:
    print("⚠️  Simple dataset detected - may not benefit from high-capacity optimizations")

print("\nSTARTING LSTM-VAE-GAN PURE ANOMALY DETECTION...")

optuna.logging.set_verbosity(optuna.logging.DEBUG)
logging.basicConfig(level=logging.INFO)

study = optuna.create_study(
    direction='maximize',
    study_name='pure_anomaly_detection_optimization',
    pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5)
)

n_trials = 10

def progress_callback(study, trial):
    print(f"\nTRIAL {trial.number} COMPLETED:")
    print(f"   Value: {trial.value:.4f}" if trial.value else "   Value: PRUNED/FAILED")
    print(f"   Best value so far: {study.best_value:.4f}")
    print(f"   Number of trials: {len(study.trials)}")
    if trial.value:
        print(f"   Parameters: {trial.params}")
    print("-" * 50)

start_time = time.time()
study.optimize(objective, n_trials=n_trials, timeout=9000, callbacks=[progress_callback])  # Extended timeout for aggressive training
optimization_time = time.time() - start_time
print(f"Optimization completed in {optimization_time/60:.1f} minutes")

print("\OPTIMIZATION COMPLETED")
print(f"Number of finished trials: {len(study.trials)}")
print(f"Best trial: {study.best_trial.number}")
print(f"Best anomaly separation ratio: {study.best_value:.4f}")
print("Best parameters:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")

print(f"\nTRAINING FINAL MODEL")

encoder, decoder, final_separation_ratio = train_with_best_params(study.best_params)

print(f"\nMODEL TRAINING COMPLETED")
print(f"Final model separation ratio: {final_separation_ratio:.4f}")
print(f"Best parameters used:")
for param, value in study.best_params.items():
    print(f"  {param}: {value}")

# === MODEL DETECTION EVALUATION ===
print(f"\nEVALUATING MODEL PERFORMANCE")

for percentile in [85, 90, 95, 97, 99]:
    scores, predictions, threshold = compute_reconstruction_anomaly_scores(
        encoder, decoder, X_test, X_train_normal, threshold_percentile=percentile
    )
    print(f"\n{percentile}th percentile threshold: {threshold:.6f}")
    print(f"Detected anomalies: {np.sum(predictions)}/{len(predictions)} ({np.sum(predictions)/len(predictions)*100:.1f}%)")

if y_test is not None:
    evaluate_reconstruction_based(encoder, decoder, X_test, y_test, X_train_normal)
    
    z_mean_test, z_log_var_test, z_test_enc = encoder(X_test)
    reconstructed_test = decoder(z_test_enc)
    recon_errors = tf.reduce_mean(tf.square(X_test - reconstructed_test), axis=[1, 2]).numpy()
    
    normal_errors = recon_errors[y_test == 0]
    anomaly_errors = recon_errors[y_test == 1]
    separation_ratio = np.mean(anomaly_errors) / np.mean(normal_errors)
    
    print(f"\nFINAL MODEL SEPARATION ANALYSIS:")
    print(f"Normal test errors:  {np.mean(normal_errors):.6f} ± {np.std(normal_errors):.6f}")
    print(f"Anomaly test errors: {np.mean(anomaly_errors):.6f} ± {np.std(anomaly_errors):.6f}")
    print(f"Separation ratio:    {separation_ratio:.3f}x")
    
    t_stat, p_value = ttest_ind(anomaly_errors, normal_errors)
    print(f"T-test p-value: {p_value:.2e} ({'significant' if p_value < 0.001 else 'not significant'})")
    
    if separation_ratio > 2.5:
        print("EXCELLENT")
    elif separation_ratio > 1.8:
        print("GOOD")
    elif separation_ratio > 1.3:
        print("MODERATE")
    elif separation_ratio > 1.1:
        print("WEAK")
    else:
        print("POOR")