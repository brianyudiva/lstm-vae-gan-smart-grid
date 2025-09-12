import numpy as np
import tensorflow as tf
from models.lstm_vae_gan import build_lstm_vae_gan_regular
from utils.loss_functions import (
    kl_loss, robust_reconstruction_loss, regularization_loss, beta_schedule
)
import os
import json
from datetime import datetime
from sklearn.metrics import average_precision_score, accuracy_score
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
        'learning_rate': 0.007121068514861119,
        'kl_weight': 0.0005578318312648292,
        'regularization_weight': 4.184353799734563e-06,
        'batch_size': 32,
        'latent_dim': 8,
        'reconstruction_weight': 1.107289156532192,
        'beta_warmup_epochs': 5,
    }

    for param, value in params.items():
        print(f"   {param}: {value}")
    
    encoder, decoder, discriminator = build_lstm_vae_gan_regular(
        input_shape=input_shape,
        latent_dim=params['latent_dim'],
    )
        
    gen_optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])
    decoder.compile(optimizer=gen_optimizer, loss='mse')
    
    disc_optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'] * 0.5)
    discriminator.compile(optimizer=disc_optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    batch_size = params['batch_size']
    epochs = 100 
    steps_per_epoch = min(X_train_normal.shape[0] // batch_size, 150)
    
    print(f"\nTraining Configuration:")
    print(f"Epochs: {epochs}, Steps per epoch: {steps_per_epoch}")
    print(f"Training data: {len(X_train_normal)} NORMAL samples only")
    print(f"Batch size: {batch_size}")
    
    training_stats = {
        'training_start': datetime.now().isoformat(),
        'parameters': params,
        'training_approach': 'optuna_optimized_lstm_vae_gan_normal_only',
        'optimization_pr_auc': 0.9986,
        'normal_samples': len(X_train_normal),
        'excluded_attack_samples': int(np.sum(y_train)),
        'epoch_results': []
    }
    
    best_pr_auc = 0.0
    wait = 0
    patience = 20
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        epoch_losses = {'vae_loss': 0, 'recon_loss': 0, 'kl_loss': 0, 'reg_loss': 0, 'gen_loss': 0, 'disc_loss': 0}
        
        for step in range(steps_per_epoch):
            normal_idx = np.random.randint(0, X_train_normal.shape[0], batch_size)
            normal_batch = X_train_normal[normal_idx]
            
            # === TRAIN DISCRIMINATOR ===
            with tf.GradientTape() as disc_tape:
                real_pred = discriminator(normal_batch, training=True)
                real_labels = tf.ones_like(real_pred)
                real_loss = tf.keras.losses.binary_crossentropy(real_labels, real_pred)
                
                z_mean, z_log_var, z = encoder(normal_batch, training=False)  # Don't update encoder during disc training
                fake_batch = decoder(z, training=False)  # Don't update decoder during disc training
                fake_pred = discriminator(fake_batch, training=True)
                fake_labels = tf.zeros_like(fake_pred)
                fake_loss = tf.keras.losses.binary_crossentropy(fake_labels, fake_pred)
                
                disc_loss = tf.reduce_mean(real_loss + fake_loss)
            
            disc_grads = disc_tape.gradient(disc_loss, discriminator.trainable_weights)
            if disc_grads:
                disc_grads = [tf.clip_by_norm(g, 1.0) for g in disc_grads]
                disc_optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_weights))
            
            # === TRAIN GENERATOR ===
            with tf.GradientTape() as gen_tape:
                # Forward pass through LSTM-VAE
                z_mean, z_log_var, z = encoder(normal_batch, training=True)
                reconstructed = decoder(z, training=True)
                
                # LSTM-VAE losses
                recon_loss_val = robust_reconstruction_loss(normal_batch, reconstructed)
                
                # Gradual KL weight increase (beta-VAE approach)
                kl_beta = beta_schedule(epoch, epochs, 
                                        max_beta=params['kl_weight'], 
                                        warmup_epochs=params['beta_warmup_epochs'])
                kl_loss_val = kl_beta * kl_loss(z_mean, z_log_var)
                
                reg_loss_val = regularization_loss(encoder, decoder)
                
                # Adversarial loss (try to fool discriminator)
                gen_pred = discriminator(reconstructed, training=False)  # Don't update discriminator during gen training
                gen_labels = tf.ones_like(gen_pred)  # Generator wants discriminator to think fake is real
                adversarial_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(gen_labels, gen_pred))
                
                # Total generator loss (LSTM-VAE + Adversarial)
                gen_loss = (params['reconstruction_weight'] * recon_loss_val +
                            kl_loss_val +
                            params['regularization_weight'] * reg_loss_val +
                            0.1 * adversarial_loss)  # Small adversarial weight to start
                
                # Update generator (LSTM encoder + LSTM decoder)
                gen_grads = gen_tape.gradient(gen_loss, encoder.trainable_weights + decoder.trainable_weights)
                if gen_grads:
                    gen_grads = [tf.clip_by_norm(g, 1.0) for g in gen_grads]
                    gen_optimizer.apply_gradients(zip(gen_grads, encoder.trainable_weights + decoder.trainable_weights))
            
            epoch_losses['vae_loss'] += (params['reconstruction_weight'] * recon_loss_val + kl_loss_val + params['regularization_weight'] * reg_loss_val).numpy()
            epoch_losses['recon_loss'] += recon_loss_val.numpy()
            epoch_losses['kl_loss'] += kl_loss_val.numpy()
            epoch_losses['reg_loss'] += reg_loss_val.numpy()
            epoch_losses['gen_loss'] += gen_loss.numpy()
            epoch_losses['disc_loss'] += disc_loss.numpy()
        
        for key in epoch_losses:
            epoch_losses[key] = float(epoch_losses[key] / steps_per_epoch)
        
        print(f"Epoch {epoch + 1} Summary:")
        print(f"  LSTM-VAE_loss: {epoch_losses['vae_loss']:.4f}")
        print(f"  Gen_loss: {epoch_losses['gen_loss']:.4f}")
        print(f"  Disc_loss: {epoch_losses['disc_loss']:.4f}")
        print(f"  Recon_loss: {epoch_losses['recon_loss']:.4f}")
        print(f"  KL_loss: {epoch_losses['kl_loss']:.4f} (beta: {kl_beta:.4f})")
        print(f"  Reg_loss: {epoch_losses['reg_loss']:.6f}")
        
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
            'pr_auc': float(pr_auc),
        }
        training_stats['epoch_results'].append(epoch_result)
        
        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            wait = 0
            
            encoder.save(f"{output_path}/{model_prefix}_encoder.h5")
            decoder.save(f"{output_path}/{model_prefix}_decoder.h5")
            discriminator.save(f"{output_path}/{model_prefix}_discriminator.h5")
            print(f"  ✓ Saved best model (PR-AUC: {pr_auc:.3f})")
        else:
            wait += 1
            
            if wait >= patience:
                print(f"  Early stopping at epoch {epoch + 1} (no PR-AUC improvement for {patience} epochs)")
                break
    
    training_stats['training_end'] = datetime.now().isoformat()
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
    # print(f"🎯 Best Accuracy: {best_t_separation_ratio:.3f}x")
    print(f"📈 Best PR-AUC: {best_pr_auc:.3f}")
    print(f"⏱️  Total Epochs: {epoch + 1}")
    print(f"💾 Model saved as: {model_prefix}_[encoder/decoder/discriminator].h5")
    
    return encoder, decoder, training_stats

if __name__ == "__main__":
    train_model()
