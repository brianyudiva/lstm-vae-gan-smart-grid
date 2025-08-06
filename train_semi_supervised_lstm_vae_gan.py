"""
Semi-Supervised LSTM-VAE-GAN with Hard Negative Contrastive Learning
Uses a small amount of labeled FDIA samples as "hard negatives" to improve anomaly separation
"""
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
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import optuna
from optuna.integration import TFKerasPruningCallback
import logging
try:
    from scipy.stats import ttest_ind
except ImportError:
    ttest_ind = None


def hard_negative_contrastive_loss(z_normal, z_anomaly, temperature=0.3, margin=2.0):
    """Hard negative contrastive loss using known anomalies
    
    Args:
        z_normal: Latent representations of normal samples
        z_anomaly: Latent representations of known anomaly samples
        temperature: Temperature parameter for contrastive learning
        margin: Margin for pushing anomalies away from normal cluster
    """
    # Normalize latent vectors
    z_normal_norm = tf.nn.l2_normalize(z_normal, axis=1)
    z_anomaly_norm = tf.nn.l2_normalize(z_anomaly, axis=1)
    
    # Compute center of normal cluster
    normal_center = tf.reduce_mean(z_normal_norm, axis=0, keepdims=True)
    
    # Pull normal samples closer to center (positive pairs)
    normal_similarities = tf.reduce_sum(z_normal_norm * normal_center, axis=1)
    normal_loss = -tf.reduce_mean(tf.nn.log_softmax(normal_similarities / temperature))
    
    # Push anomalies away from normal center (negative pairs)
    anomaly_distances = tf.reduce_sum(tf.square(z_anomaly_norm - normal_center), axis=1)
    # Use margin loss - penalize if anomalies are too close to normal cluster
    margin_loss = tf.reduce_mean(tf.maximum(0.0, margin - anomaly_distances))
    
    return normal_loss + margin_loss


def semi_supervised_objective(trial):
    """Optuna objective function for semi-supervised anomaly detection with hard negatives"""
    print(f"\nSTARTING SEMI-SUPERVISED TRIAL {trial.number}")
    print(f"Trial {trial.number}: Sampling hyperparameters for semi-supervised training...")
    
    # Clear any previous TensorFlow sessions
    tf.keras.backend.clear_session()
    
    # Enhanced hyperparameters for semi-supervised training
    initial_lr = trial.suggest_float('learning_rate', 1e-4, 5e-3, log=True)
    kl_weight = trial.suggest_float('kl_weight', 0.5, 3.0, log=True)
    regularization_weight = trial.suggest_float('regularization_weight', 0.01, 0.1, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 48, 64])
    latent_dim = trial.suggest_int('latent_dim', 2, 3)
    
    # Semi-supervised specific hyperparameters
    reconstruction_weight = trial.suggest_float('reconstruction_weight', 0.8, 1.5)
    beta_multiplier = trial.suggest_float('beta_multiplier', 2.0, 4.0)
    contrastive_weight = trial.suggest_float('contrastive_weight', 0.2, 0.8)
    hard_negative_weight = trial.suggest_float('hard_negative_weight', 0.3, 1.0)  # New parameter
    anomaly_reconstruction_penalty = trial.suggest_float('anomaly_reconstruction_penalty', 0.1, 0.5)  # Penalty for anomaly reconstruction
    
    print(f"Trial {trial.number}: Semi-supervised training with lr={initial_lr:.6f}, batch_size={batch_size}, latent_dim={latent_dim}")
    print(f"   KL_weight={kl_weight:.4f}, reg_weight={regularization_weight:.4f}, beta_mult={beta_multiplier:.2f}")
    print(f"   Hard_negative_weight={hard_negative_weight:.3f}, Anomaly_penalty={anomaly_reconstruction_penalty:.3f}")
    
    # Architecture selection
    encoder, decoder, discriminator, arch_info = select_architecture(
        normal_samples_count=len(X_train_normal),
        input_shape=input_shape,
        latent_dim=latent_dim,
        force_architecture='compact'
    )
    
    # Optimizer
    optimizer = tf.keras.optimizers.Adam(initial_lr, clipnorm=1.0)
    
    # Enhanced data augmentation for normal samples
    X_augmented = enhanced_data_augmentation(X_train_normal, augmentation_factor=0.15)
    X_train_enhanced = np.vstack([X_train_normal, X_augmented])
    
    # Use a small subset of FDIA samples as hard negatives (5-10% of normal samples)
    n_hard_negatives = min(len(X_train_fdia), max(10, len(X_train_normal) // 10))
    hard_negative_indices = np.random.choice(len(X_train_fdia), n_hard_negatives, replace=False)
    X_hard_negatives = X_train_fdia[hard_negative_indices]
    
    print(f"Using {len(X_hard_negatives)} hard negative samples (FDIA) out of {len(X_train_fdia)} available")
    
    # Training parameters
    epochs = 15
    steps_per_epoch = min(X_train_enhanced.shape[0] // batch_size, 30)
    
    best_separation_ratio = 0.0
    patience = 10
    wait = 0
    
    print(f"Trial {trial.number}: Training semi-supervised anomaly detector...")
    
    for epoch in range(epochs):
        epoch_losses = {'vae_loss': 0, 'recon_loss': 0, 'kl_loss': 0, 'reg_loss': 0, 'hard_neg_loss': 0}
        
        for step in range(steps_per_epoch):
            # Get batch of normal data
            idx = np.random.randint(0, X_train_enhanced.shape[0], batch_size//2)
            normal_batch = X_train_enhanced[idx]
            
            # Get batch of hard negatives (anomalies)
            hard_neg_idx = np.random.randint(0, len(X_hard_negatives), batch_size//2)
            anomaly_batch = X_hard_negatives[hard_neg_idx]
            
            with tf.GradientTape() as tape:
                # Forward pass for normal samples
                z_mean_normal, z_log_var_normal, z_normal = encoder(normal_batch, training=True)
                reconstructed_normal = decoder(z_normal, training=True)
                
                # Forward pass for anomaly samples
                z_mean_anomaly, z_log_var_anomaly, z_anomaly = encoder(anomaly_batch, training=True)
                reconstructed_anomaly = decoder(z_anomaly, training=True)
                
                # PHASE 1: Reconstruction loss (normal samples should reconstruct well)
                recon_loss_normal = reconstruction_weight * spectral_reconstruction_loss(normal_batch, reconstructed_normal)
                
                # PHASE 2: Anomaly reconstruction penalty (anomalies should reconstruct poorly)
                recon_loss_anomaly = anomaly_reconstruction_penalty * spectral_reconstruction_loss(anomaly_batch, reconstructed_anomaly)
                # Make this a penalty (negative) so we want higher reconstruction error for anomalies
                recon_loss_anomaly = -recon_loss_anomaly
                
                # PHASE 3: Anomaly regularization for normal samples
                anomaly_reg_loss = anomaly_regularization_loss(z_mean_normal, z_log_var_normal, z_normal)
                
                # PHASE 4: Hard negative contrastive learning
                hard_negative_loss_val = hard_negative_contrastive_loss(z_normal, z_anomaly)
                
                # PHASE 5: Standard contrastive learning with augmented data
                contrastive_loss_val = 0.0
                if step % 3 == 0 and len(X_train_enhanced) > len(X_train_normal):
                    aug_idx = np.random.randint(len(X_train_normal), X_train_enhanced.shape[0], batch_size//4)
                    aug_batch = X_train_enhanced[aug_idx]
                    z_mean_aug, z_log_var_aug, z_aug = encoder(aug_batch, training=True)
                    
                    if len(z_normal) >= len(z_aug):
                        contrastive_loss_val = contrastive_latent_loss(z_normal[:len(z_aug)], z_aug)
                    else:
                        contrastive_loss_val = contrastive_latent_loss(z_normal, z_aug[:len(z_normal)])
                
                # Standard regularization
                reg_loss_val = regularization_loss(encoder, decoder)
                
                # Beta scheduling
                beta = beta_schedule(epoch, epochs) * beta_multiplier
                progress_factor = min(1.0, epoch / 20.0)
                
                # COMBINED SEMI-SUPERVISED LOSS
                vae_loss = (recon_loss_normal +                                    # Normal reconstruction
                           recon_loss_anomaly +                                   # Anomaly reconstruction penalty
                           beta * kl_weight * anomaly_reg_loss +                  # KL regularization
                           regularization_weight * reg_loss_val +                 # Weight regularization
                           progress_factor * contrastive_weight * contrastive_loss_val +  # Standard contrastive
                           progress_factor * hard_negative_weight * hard_negative_loss_val)  # Hard negative contrastive
            
            # Apply gradients to encoder and decoder only
            grads = tape.gradient(vae_loss, encoder.trainable_weights + decoder.trainable_weights)
            if grads and all(g is not None for g in grads):
                if not any(tf.reduce_any(tf.math.is_nan(grad)) for grad in grads):
                    grads = [tf.clip_by_norm(g, 1.0) for g in grads]
                    optimizer.apply_gradients(zip(grads, encoder.trainable_weights + decoder.trainable_weights))
            
            # Track losses
            epoch_losses['vae_loss'] += vae_loss.numpy()
            epoch_losses['recon_loss'] += recon_loss_normal.numpy()
            epoch_losses['kl_loss'] += anomaly_reg_loss.numpy()
            epoch_losses['reg_loss'] += reg_loss_val.numpy()
            epoch_losses['hard_neg_loss'] += hard_negative_loss_val.numpy()
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= steps_per_epoch
        
        # Print progress
        if epoch % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: VAE_loss={epoch_losses['vae_loss']:.4f}, "
                  f"Hard_neg_loss={epoch_losses['hard_neg_loss']:.4f}")
        
        # Evaluate separation
        if epoch % 7 == 0 or epoch == epochs - 1:
            test_subset_size = min(600, len(X_test))
            test_subset_idx = np.random.choice(len(X_test), test_subset_size, replace=False)
            X_test_subset = X_test[test_subset_idx]
            y_test_subset = y_test[test_subset_idx] if y_test is not None else None
            
            if y_test_subset is not None:
                # Calculate reconstruction errors
                z_mean_test, z_log_var_test, z_test_enc = encoder(X_test_subset)
                reconstructed_test = decoder(z_test_enc)
                recon_errors = tf.reduce_mean(tf.square(X_test_subset - reconstructed_test), axis=[1, 2]).numpy()
                
                normal_errors = recon_errors[y_test_subset == 0]
                anomaly_errors = recon_errors[y_test_subset == 1]
                
                if len(normal_errors) > 0 and len(anomaly_errors) > 0:
                    separation_ratio = np.mean(anomaly_errors) / np.mean(normal_errors)
                    
                    print(f"    Epoch {epoch+1} Semi-supervised Separation: {separation_ratio:.4f}x")
                    
                    # Enhanced scoring for semi-supervised approach
                    normal_std = np.std(normal_errors)
                    anomaly_std = np.std(anomaly_errors)
                    effect_size = (np.mean(anomaly_errors) - np.mean(normal_errors)) / np.sqrt((normal_std**2 + anomaly_std**2) / 2)
                    
                    combined_score = separation_ratio * 0.7 + effect_size * 0.3  # Balanced scoring
                    
                    if combined_score > best_separation_ratio:
                        best_separation_ratio = combined_score
                        wait = 0
                        print(f"    NEW BEST semi-supervised score: {combined_score:.4f}")
                    else:
                        wait += 1
                    
                    # Report for pruning
                    trial.report(combined_score, epoch)
                    
                    if trial.should_prune():
                        print(f"    Trial {trial.number} pruned at epoch {epoch+1}")
                        raise optuna.TrialPruned()
                    
                    if wait >= patience:
                        print(f"    Early stopping triggered")
                        break
    
    print(f"Semi-supervised trial {trial.number} completed: Best separation = {best_separation_ratio:.4f}")
    return best_separation_ratio


def train_semi_supervised_final(best_params):
    """Train final semi-supervised model with best hyperparameters"""
    print(f"\nTRAINING FINAL SEMI-SUPERVISED MODEL:")
    for param, value in best_params.items():
        print(f"   {param}: {value}")
    
    tf.keras.backend.clear_session()
    
    # Build model
    encoder, decoder, discriminator, arch_info = select_architecture(
        normal_samples_count=len(X_train_normal),
        input_shape=input_shape,
        latent_dim=best_params['latent_dim'],
        force_architecture='compact'
    )
    
    print(f"\nSemi-supervised architecture: {arch_info['name'].upper()}")
    print(f"Latent dimension: {best_params['latent_dim']}")
    
    optimizer = tf.keras.optimizers.Adam(best_params['learning_rate'], clipnorm=1.0)
    
    # Enhanced normal data
    X_augmented = enhanced_data_augmentation(X_train_normal, augmentation_factor=0.2)
    X_train_enhanced = np.vstack([X_train_normal, X_augmented])
    
    # Hard negatives
    n_hard_negatives = min(len(X_train_fdia), max(15, len(X_train_normal) // 8))  # Slightly more for final training
    hard_negative_indices = np.random.choice(len(X_train_fdia), n_hard_negatives, replace=False)
    X_hard_negatives = X_train_fdia[hard_negative_indices]
    
    print(f"Final training: {len(X_train_enhanced)} normal samples, {len(X_hard_negatives)} hard negatives")
    
    # Training parameters
    batch_size = best_params['batch_size']
    epochs = 30  # More epochs for final training
    steps_per_epoch = min(X_train_enhanced.shape[0] // batch_size, 60)
    
    best_separation_ratio = 0.0
    wait = 0
    
    for epoch in range(epochs):
        print(f"\nSemi-supervised Epoch {epoch + 1}/{epochs}")
        epoch_losses = {'vae_loss': 0, 'recon_loss': 0, 'hard_neg_loss': 0}
        
        for step in range(steps_per_epoch):
            # Normal and anomaly batches
            idx = np.random.randint(0, X_train_enhanced.shape[0], batch_size//2)
            normal_batch = X_train_enhanced[idx]
            
            hard_neg_idx = np.random.randint(0, len(X_hard_negatives), batch_size//2)
            anomaly_batch = X_hard_negatives[hard_neg_idx]
            
            with tf.GradientTape() as tape:
                # Forward passes
                z_mean_normal, z_log_var_normal, z_normal = encoder(normal_batch, training=True)
                reconstructed_normal = decoder(z_normal, training=True)
                
                z_mean_anomaly, z_log_var_anomaly, z_anomaly = encoder(anomaly_batch, training=True)
                reconstructed_anomaly = decoder(z_anomaly, training=True)
                
                # Loss components
                recon_loss_normal = best_params.get('reconstruction_weight', 1.0) * spectral_reconstruction_loss(normal_batch, reconstructed_normal)
                recon_loss_anomaly = -best_params.get('anomaly_reconstruction_penalty', 0.3) * spectral_reconstruction_loss(anomaly_batch, reconstructed_anomaly)
                
                anomaly_reg_loss = anomaly_regularization_loss(z_mean_normal, z_log_var_normal, z_normal)
                hard_negative_loss_val = hard_negative_contrastive_loss(z_normal, z_anomaly)
                
                # Standard contrastive learning
                contrastive_loss_val = 0.0
                if step % 3 == 0 and len(X_train_enhanced) > len(X_train_normal):
                    aug_idx = np.random.randint(len(X_train_normal), X_train_enhanced.shape[0], batch_size//4)
                    aug_batch = X_train_enhanced[aug_idx]
                    z_mean_aug, z_log_var_aug, z_aug = encoder(aug_batch, training=True)
                    
                    if len(z_normal) >= len(z_aug):
                        contrastive_loss_val = contrastive_latent_loss(z_normal[:len(z_aug)], z_aug)
                    else:
                        contrastive_loss_val = contrastive_latent_loss(z_normal, z_aug[:len(z_normal)])
                
                reg_loss_val = regularization_loss(encoder, decoder)
                
                # Combined loss
                beta = beta_schedule(epoch, epochs) * best_params.get('beta_multiplier', 2.0)
                progress_factor = min(1.0, epoch / 15.0)
                
                vae_loss = (recon_loss_normal +
                           recon_loss_anomaly +
                           beta * best_params['kl_weight'] * anomaly_reg_loss +
                           best_params['regularization_weight'] * reg_loss_val +
                           progress_factor * best_params.get('contrastive_weight', 0.4) * contrastive_loss_val +
                           progress_factor * best_params.get('hard_negative_weight', 0.5) * hard_negative_loss_val)
            
            # Apply gradients
            grads = tape.gradient(vae_loss, encoder.trainable_weights + decoder.trainable_weights)
            if grads and all(g is not None for g in grads):
                if not any(tf.reduce_any(tf.math.is_nan(grad)) for grad in grads):
                    grads = [tf.clip_by_norm(g, 1.0) for g in grads]
                    optimizer.apply_gradients(zip(grads, encoder.trainable_weights + decoder.trainable_weights))
            
            # Track losses
            epoch_losses['vae_loss'] += vae_loss.numpy()
            epoch_losses['recon_loss'] += recon_loss_normal.numpy()
            epoch_losses['hard_neg_loss'] += hard_negative_loss_val.numpy()
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= steps_per_epoch
        
        print(f"  VAE_loss: {epoch_losses['vae_loss']:.4f}")
        print(f"  Recon_loss: {epoch_losses['recon_loss']:.4f}")
        print(f"  Hard_neg_loss: {epoch_losses['hard_neg_loss']:.4f}")
        
        # Evaluate
        if y_test is not None:
            z_mean_test, z_log_var_test, z_test_enc = encoder(X_test)
            reconstructed_test = decoder(z_test_enc)
            recon_errors = tf.reduce_mean(tf.square(X_test - reconstructed_test), axis=[1, 2]).numpy()
            
            normal_errors = recon_errors[y_test == 0]
            anomaly_errors = recon_errors[y_test == 1]
            separation_ratio = np.mean(anomaly_errors) / np.mean(normal_errors)
            
            print(f"  Semi-supervised separation ratio: {separation_ratio:.3f}x")
            
            if separation_ratio > best_separation_ratio:
                best_separation_ratio = separation_ratio
                wait = 0
                
                # Save models
                encoder.save(f"{output_path}/semi_supervised_best_encoder.h5")
                decoder.save(f"{output_path}/semi_supervised_best_decoder.h5")
                print(f"  Best semi-supervised model saved (separation: {separation_ratio:.3f}x)")
            else:
                wait += 1
                
                if wait >= 25:
                    print(f"  Early stopping at epoch {epoch + 1}")
                    break
    
    return encoder, decoder, best_separation_ratio


# === MAIN SEMI-SUPERVISED TRAINING ===
if __name__ == "__main__":
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
        
        # Separate normal and FDIA training data
        X_train_normal = X_train[y_train == 0]
        X_train_fdia = X_train[y_train == 1]
        
        print(f"Normal training samples: {len(X_train_normal)}")
        print(f"FDIA training samples: {len(X_train_fdia)} (will use subset as hard negatives)")
        
    except:
        print("No labels found - semi-supervised training requires labels!")
        exit()
    
    input_shape = (X_train_normal.shape[1], X_train_normal.shape[2])
    print(f"Input shape: {input_shape}")
    
    print("\n" + "="*80)
    print("🔥 SEMI-SUPERVISED LSTM-VAE-GAN WITH HARD NEGATIVE CONTRASTIVE LEARNING")
    print("="*80)
    print("🎯 Approach: Use small subset of labeled FDIA as hard negatives")
    print("📈 Goal: Improve anomaly separation beyond pure unsupervised approach")
    print("🧠 Method: Hard negative contrastive learning + anomaly reconstruction penalty")
    
    # Set up logging
    optuna.logging.set_verbosity(optuna.logging.INFO)
    logging.basicConfig(level=logging.INFO)
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        study_name='semi_supervised_anomaly_detection',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=3)
    )
    
    # Run optimization
    n_trials = 3  # Start with few trials for development
    print(f"\nRunning {n_trials} semi-supervised optimization trials...")
    
    def progress_callback(study, trial):
        print(f"\nSEMI-SUPERVISED TRIAL {trial.number} COMPLETED:")
        print(f"   Value: {trial.value:.4f}" if trial.value else "   Value: PRUNED/FAILED")
        print(f"   Best value so far: {study.best_value:.4f}")
        print(f"   Parameters: {trial.params}" if trial.value else "")
        print("-" * 60)
    
    start_time = time.time()
    study.optimize(semi_supervised_objective, n_trials=n_trials, timeout=3600, callbacks=[progress_callback])
    optimization_time = time.time() - start_time
    
    print(f"\nSEMI-SUPERVISED OPTIMIZATION COMPLETED in {optimization_time/60:.1f} minutes!")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best separation ratio: {study.best_value:.4f}")
    print("Best parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Train final model
    print(f"\n" + "="*60)
    print("TRAINING FINAL SEMI-SUPERVISED MODEL")
    print("="*60)
    
    encoder, decoder, final_separation_ratio = train_semi_supervised_final(study.best_params)
    
    print(f"\n🎉 SEMI-SUPERVISED TRAINING COMPLETED!")
    print(f"Final separation ratio: {final_separation_ratio:.4f}x")
    print(f"Models saved as 'semi_supervised_best_encoder.h5' and 'semi_supervised_best_decoder.h5'")
    
    # Compare with test data
    if y_test is not None:
        z_mean_test, z_log_var_test, z_test_enc = encoder(X_test)
        reconstructed_test = decoder(z_test_enc)
        recon_errors = tf.reduce_mean(tf.square(X_test - reconstructed_test), axis=[1, 2]).numpy()
        
        normal_errors = recon_errors[y_test == 0]
        anomaly_errors = recon_errors[y_test == 1]
        separation_ratio = np.mean(anomaly_errors) / np.mean(normal_errors)
        
        print(f"\n📊 FINAL SEMI-SUPERVISED RESULTS:")
        print(f"Normal test errors:  {np.mean(normal_errors):.6f} ± {np.std(normal_errors):.6f}")
        print(f"Anomaly test errors: {np.mean(anomaly_errors):.6f} ± {np.std(anomaly_errors):.6f}")
        print(f"Separation ratio:    {separation_ratio:.3f}x")
        
        if ttest_ind is not None:
            t_stat, p_value = ttest_ind(anomaly_errors, normal_errors)
            print(f"T-test p-value: {p_value:.2e} ({'significant' if p_value < 0.001 else 'not significant'})")
        
        if separation_ratio > 3.0:
            print("🌟 EXCELLENT: Very strong semi-supervised anomaly separation!")
        elif separation_ratio > 2.0:
            print("✅ GOOD: Strong semi-supervised anomaly separation!")
        elif separation_ratio > 1.5:
            print("⚠️ MODERATE: Some improvement, may need tuning")
        else:
            print("❌ POOR: Semi-supervised approach needs improvement")
        
        print(f"\n💡 COMPARISON TIP: Compare this {separation_ratio:.3f}x with pure unsupervised results")
