import numpy as np
import tensorflow as tf
from models.lstm_vae_gan import build_lstm_vae_gan_regular
from utils.loss_functions import (
    kl_loss, robust_reconstruction_loss, regularization_loss, beta_schedule
)
import os
import json
from datetime import datetime
from sklearn.metrics import average_precision_score
from utils.utils import convert_to_json_serializable

def train_model():    
    sequence_path = "data/sequences"
    output_path = "outputs/checkpoints"
    model_prefix = "lstm_vae_gan"
    os.makedirs(output_path, exist_ok=True)

    X_train = np.load(f"{sequence_path}/X_train.npy")
    X_test = np.load(f"{sequence_path}/X_test.npy")
    y_train = np.load(f"{sequence_path}/y_train_binary.npy")
    y_test = np.load(f"{sequence_path}/y_test_binary.npy")

    print(f"Labels loaded - Train FDIA: {np.sum(y_train)}/{len(y_train)}, Test FDIA: {np.sum(y_test)}/{len(y_test)}")
    
    X_train_normal = X_train[y_train == 0]
    print(f"Training on normal data: {len(X_train_normal)} samples")
    print(f"Excluded {np.sum(y_train)} attack samples from training")

    input_shape = (X_train_normal.shape[1], X_train_normal.shape[2])
    
    tf.keras.backend.clear_session()
    
    params = {
        'learning_rate': 0.001,             # Standard learning rate
        'kl_weight': 0.005,                 # Lower KL weight since attacks are now very strong
        'regularization_weight': 1e-6,     # Light regularization
        'batch_size': 128,                  # Larger batch for stable training
        'latent_dim': 16,                   # Smaller latent space for tighter normal representation
        'reconstruction_weight': 1.0,      # Focus on reconstruction
        'beta_warmup_epochs': 5,            # Faster warmup since attacks are obvious
    }

    print("\nParameters:")
    for param, value in params.items():
        print(f"   {param}: {value}")
    
    encoder, decoder, _ = build_lstm_vae_gan_regular(
        input_shape=input_shape,
        latent_dim=params['latent_dim'],
    )
    
    optimizer = tf.keras.optimizers.Adam(params['learning_rate'], clipnorm=1.0)
    
    batch_size = params['batch_size']
    epochs = 100  # Fewer epochs since attacks are now very obvious
    steps_per_epoch = min(X_train_normal.shape[0] // batch_size, 150)  # Fewer steps since larger batches
    
    print(f"\nTraining Configuration:")
    print(f"Epochs: {epochs}, Steps per epoch: {steps_per_epoch}")
    print(f"Training data: {len(X_train_normal)} NORMAL samples only")
    print(f"Batch size: {batch_size}")
    
    training_stats = {
        'training_start': datetime.now().isoformat(),
        'parameters': params,
        'training_approach': 'normal_only_vae',
        'normal_samples': len(X_train_normal),
        'excluded_attack_samples': int(np.sum(y_train)),
        'epoch_results': []
    }
    
    best_separation_ratio = 0.0
    best_pr_auc = 0.0
    wait = 0
    patience = 20  # Higher patience for normal-only training
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        epoch_losses = {'vae_loss': 0, 'recon_loss': 0, 'kl_loss': 0, 'reg_loss': 0}
        
        for _ in range(steps_per_epoch):
            # Sample normal batch only
            normal_idx = np.random.randint(0, X_train_normal.shape[0], batch_size)
            normal_batch = X_train_normal[normal_idx]
            
            with tf.GradientTape() as tape:
                # Forward pass through VAE
                z_mean, z_log_var, z = encoder(normal_batch, training=True)
                reconstructed = decoder(z, training=True)
                
                # Compute losses
                recon_loss_val = robust_reconstruction_loss(normal_batch, reconstructed)
                
                # Gradual KL weight increase (beta-VAE approach)
                kl_beta = beta_schedule(epoch, epochs, 
                                      max_beta=params['kl_weight'], 
                                      warmup_epochs=params['beta_warmup_epochs'])
                kl_loss_val = kl_beta * kl_loss(z_mean, z_log_var)
                
                reg_loss_val = regularization_loss(encoder, decoder)
                
                # Total VAE loss - focused on normal data reconstruction
                vae_loss = (params['reconstruction_weight'] * recon_loss_val +
                           kl_loss_val +
                           params['regularization_weight'] * reg_loss_val)
            
            # Update VAE parameters
            grads = tape.gradient(vae_loss, encoder.trainable_weights + decoder.trainable_weights)
            if grads:
                grads = [tf.clip_by_norm(g, 1.0) for g in grads]
                optimizer.apply_gradients(zip(grads, encoder.trainable_weights + decoder.trainable_weights))
            
            # Track losses
            epoch_losses['vae_loss'] += vae_loss.numpy()
            epoch_losses['recon_loss'] += recon_loss_val.numpy()
            epoch_losses['kl_loss'] += kl_loss_val.numpy()
            epoch_losses['reg_loss'] += reg_loss_val.numpy()
        
        # Average losses over epoch
        for key in epoch_losses:
            epoch_losses[key] = float(epoch_losses[key] / steps_per_epoch)
        
        print(f"Epoch {epoch + 1} Summary:")
        print(f"  VAE_loss: {epoch_losses['vae_loss']:.4f}")
        print(f"  Recon_loss: {epoch_losses['recon_loss']:.4f}")
        print(f"  KL_loss: {epoch_losses['kl_loss']:.4f} (beta: {kl_beta:.4f})")
        print(f"  Reg_loss: {epoch_losses['reg_loss']:.6f}")
        
        # Evaluate on test set (both normal and attacks)
        _, _, z_test_enc = encoder(X_test)
        reconstructed_test = decoder(z_test_enc)
        recon_errors = tf.reduce_mean(tf.square(X_test - reconstructed_test), axis=[1, 2]).numpy()
        
        # Separation analysis
        normal_errors = recon_errors[y_test == 0]
        attack_errors = recon_errors[y_test == 1]
        separation_ratio = float(np.mean(attack_errors) / np.mean(normal_errors))
        
        # Performance metrics
        pr_auc = float(average_precision_score(y_test, recon_errors))
        
        # Normal data reconstruction quality (should be low)
        normal_recon_quality = float(np.mean(normal_errors))
        attack_recon_quality = float(np.mean(attack_errors))
        
        print(f"  Normal reconstruction error: {normal_recon_quality:.6f}")
        print(f"  Attack reconstruction error: {attack_recon_quality:.6f}")
        print(f"  Separation ratio: {separation_ratio:.3f}x")
        print(f"  PR AUC: {pr_auc:.3f}")
        
        # Store epoch results
        epoch_result = {
            'epoch': epoch + 1,
            'losses': {k: float(v) for k, v in epoch_losses.items()},
            'kl_beta': float(kl_beta),
            'normal_recon_error': normal_recon_quality,
            'attack_recon_error': attack_recon_quality,
            'separation_ratio': float(separation_ratio),
            'pr_auc': float(pr_auc)
        }
        training_stats['epoch_results'].append(epoch_result)
        
        # Early stopping based on PR-AUC
        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            best_separation_ratio = separation_ratio
            wait = 0
            
            # Save best model
            encoder.save(f"{output_path}/{model_prefix}_encoder.h5")
            decoder.save(f"{output_path}/{model_prefix}_decoder.h5")
            print(f"  ✓ Saved best model (PR-AUC: {pr_auc:.3f})")
        else:
            wait += 1
            
            if wait >= patience:
                print(f"  Early stopping at epoch {epoch + 1} (no PR-AUC improvement for {patience} epochs)")
                break
    
    # Final statistics
    training_stats['training_end'] = datetime.now().isoformat()
    training_stats['best_separation_ratio'] = float(best_separation_ratio)
    training_stats['best_pr_auc'] = float(best_pr_auc)
    training_stats['total_epochs'] = epoch + 1
    training_stats['early_stopped'] = wait >= patience
    
    # Save training statistics
    serializable_stats = convert_to_json_serializable(training_stats)
    
    with open(f"{output_path}/{model_prefix}_stats.json", 'w') as f:
        json.dump(serializable_stats, f, indent=2)
    
    print(f"\n" + "="*60)
    print(f"TRAINING COMPLETED!")
    print(f"="*60)
    print(f"📊 Best Separation Ratio: {best_separation_ratio:.3f}x")
    print(f"📈 Best PR-AUC: {best_pr_auc:.3f}")
    print(f"⏱️  Total Epochs: {epoch + 1}")
    print(f"💾 Model saved as: {model_prefix}_[encoder/decoder].h5")
    
    return encoder, decoder, training_stats

if __name__ == "__main__":
    train_model()
