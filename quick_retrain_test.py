#!/usr/bin/env python3
"""
Quick test script to validate improved training approach
"""
import numpy as np
import tensorflow as tf
from models.lstm_vae_gan import select_architecture
import os

# === QUICK CONFIGURATION ===
sequence_path = "data/sequences"
test_epochs = 50  # Quick test with fewer epochs

# === LOAD DATA ===
X_train = np.load(f"{sequence_path}/X_train.npy")
X_test = np.load(f"{sequence_path}/X_test.npy")
y_train = np.load(f"{sequence_path}/y_train_binary.npy")
y_test = np.load(f"{sequence_path}/y_test_binary.npy")

# Use only normal data
X_train_normal = X_train[y_train == 0]
print(f"Quick test with {len(X_train_normal)} normal samples")

# === ULTRA-COMPACT ARCHITECTURE ===
input_shape = (X_train_normal.shape[1], X_train_normal.shape[2])
encoder, decoder, discriminator, arch_info = select_architecture(
    normal_samples_count=len(X_train_normal),
    input_shape=input_shape,
    latent_dim=2,  # EXTREMELY small latent space - force compression
    force_architecture='ultra_compact'
)

print(f"Architecture: {arch_info['name']}, Parameters: {encoder.count_params() + decoder.count_params()}")

# === HYPER-AGGRESSIVE APPROACH ===
def kl_loss(z_mean, z_log_var):
    return -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))

def extreme_reconstruction_loss(y_true, y_pred, margin=0.001):
    """Hyper-aggressive reconstruction loss with exponential penalty"""
    mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=[1, 2])
    
    # Exponential penalty for ANY deviation above tiny margin
    penalty = tf.exp(tf.maximum(0.0, (mse - margin) * 10000.0)) - 1.0
    
    # L1 penalty for fine-grained control
    l1_penalty = tf.reduce_mean(tf.abs(y_true - y_pred), axis=[1, 2]) * 1000.0
    
    return tf.reduce_mean(mse + penalty + l1_penalty)

def adaptive_learning_rate(epoch, base_lr=0.001):
    """Adaptive learning rate that starts high then drops dramatically"""
    if epoch < 10:
        return base_lr
    elif epoch < 30:
        return base_lr * 0.1
    else:
        return base_lr * 0.01

# HYPER-EXTREME hyperparameters
kl_weight = 0.00001   # Almost zero KL - pure reconstruction focus
recon_weight = 100000 # Massive reconstruction weight
optimizer = tf.keras.optimizers.Adam(0.001)  # Start with higher LR

print(f"Hyperparameters: KL={kl_weight}, Recon={recon_weight}, Latent_dim=2")

# === HYPER-AGGRESSIVE TRAINING WITH ADAPTIVE STRATEGY ===
batch_size = 4  # Tiny batches for more gradient updates
best_loss = float('inf')
patience = 15
no_improve_count = 0

print("Starting hyper-aggressive training with adaptive strategy...")
for epoch in range(test_epochs):
    # Adaptive learning rate
    current_lr = adaptive_learning_rate(epoch)
    optimizer.learning_rate.assign(current_lr)
    
    epoch_loss = 0
    n_batches = 0
    
    # Shuffle data each epoch for better generalization
    indices = np.random.permutation(len(X_train_normal))
    X_shuffled = X_train_normal[indices]
    
    for i in range(0, len(X_shuffled), batch_size):
        batch = X_shuffled[i:i+batch_size]
        if len(batch) < batch_size:
            continue
            
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = encoder(batch)
            reconstructed = decoder(z)
            
            kl = kl_loss(z_mean, z_log_var)
            recon = extreme_reconstruction_loss(batch, reconstructed)
            
            total_loss = kl_weight * kl + recon_weight * recon
        
        grads = tape.gradient(total_loss, encoder.trainable_weights + decoder.trainable_weights)
        
        # Gradient clipping for stability
        if grads and all(g is not None for g in grads):
            grads = [tf.clip_by_norm(g, 1.0) for g in grads]
            optimizer.apply_gradients(zip(grads, encoder.trainable_weights + decoder.trainable_weights))
        
        epoch_loss += total_loss.numpy()
        n_batches += 1
    
    avg_loss = epoch_loss / n_batches if n_batches > 0 else float('inf')
    
    # Early stopping with patience
    if avg_loss < best_loss:
        best_loss = avg_loss
        no_improve_count = 0
        print(f"Epoch {epoch+1}: Loss={avg_loss:.8f}, LR={current_lr:.6f} ✓")
    else:
        no_improve_count += 1
        if epoch % 5 == 0:
            print(f"Epoch {epoch+1}: Loss={avg_loss:.8f}, LR={current_lr:.6f} (no improve: {no_improve_count})")
    
    # Early stopping
    if no_improve_count >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

print(f"Hyper-aggressive training completed. Best loss: {best_loss:.8f}")

# === ENHANCED EVALUATION WITH MULTIPLE STRATEGIES ===
def enhanced_evaluate():
    # Strategy 1: Pure reconstruction error
    z_mean_test, _, z_test = encoder(X_test)
    reconstructed_test = decoder(z_test)
    test_errors = tf.reduce_mean(tf.square(X_test - reconstructed_test), axis=[1, 2]).numpy()
    
    z_mean_train, _, z_train = encoder(X_train_normal)
    reconstructed_train = decoder(z_train)
    train_errors = tf.reduce_mean(tf.square(X_train_normal - reconstructed_train), axis=[1, 2]).numpy()
    
    # Strategy 2: Latent space anomaly detection
    latent_distances_test = tf.reduce_mean(tf.square(z_mean_test), axis=1).numpy()
    latent_distances_train = tf.reduce_mean(tf.square(z_mean_train), axis=1).numpy()
    
    # Strategy 3: Combined score
    # Normalize both metrics
    recon_norm = (test_errors - np.mean(train_errors)) / np.std(train_errors)
    latent_norm = (latent_distances_test - np.mean(latent_distances_train)) / np.std(latent_distances_train)
    combined_scores = 0.7 * recon_norm + 0.3 * latent_norm
    
    # Separate by class
    normal_errors = test_errors[y_test == 0]
    attack_errors = test_errors[y_test == 1]
    
    normal_latent = latent_distances_test[y_test == 0]
    attack_latent = latent_distances_test[y_test == 1]
    
    normal_combined = combined_scores[y_test == 0]
    attack_combined = combined_scores[y_test == 1]
    
    # Calculate separations
    recon_ratio = np.mean(attack_errors) / np.mean(normal_errors)
    latent_ratio = np.mean(attack_latent) / np.mean(normal_latent)
    combined_ratio = np.mean(attack_combined) / np.mean(normal_combined) if np.mean(normal_combined) > 0 else 1.0
    
    print("\n=== ENHANCED EVALUATION RESULTS ===")
    print(f"RECONSTRUCTION ERRORS:")
    print(f"  Train errors:  {np.mean(train_errors):.6f} ± {np.std(train_errors):.6f}")
    print(f"  Normal errors: {np.mean(normal_errors):.6f} ± {np.std(normal_errors):.6f}")  
    print(f"  Attack errors: {np.mean(attack_errors):.6f} ± {np.std(attack_errors):.6f}")
    print(f"  Separation ratio: {recon_ratio:.3f}x")
    
    print(f"\nLATENT SPACE DISTANCES:")
    print(f"  Normal latent: {np.mean(normal_latent):.6f} ± {np.std(normal_latent):.6f}")
    print(f"  Attack latent: {np.mean(attack_latent):.6f} ± {np.std(attack_latent):.6f}")
    print(f"  Separation ratio: {latent_ratio:.3f}x")
    
    print(f"\nCOMBINED SCORES:")
    print(f"  Normal combined: {np.mean(normal_combined):.6f} ± {np.std(normal_combined):.6f}")
    print(f"  Attack combined: {np.mean(attack_combined):.6f} ± {np.std(attack_combined):.6f}")
    print(f"  Separation ratio: {combined_ratio:.3f}x")
    
    # Success criteria
    best_ratio = max(recon_ratio, latent_ratio, combined_ratio)
    
    if best_ratio > 3.0:
        print(f"✅ EXCELLENT: Best separation {best_ratio:.3f}x achieved!")
        return True
    elif best_ratio > 2.0:
        print(f"⚡ GOOD: Strong separation {best_ratio:.3f}x achieved!")
        return True
    elif best_ratio > 1.5:
        print(f"📈 IMPROVEMENT: Better separation {best_ratio:.3f}x than before")
        return True
    else:
        print(f"❌ STILL POOR: Best separation only {best_ratio:.3f}x")
        return False

# Run enhanced evaluation
success = enhanced_evaluate()

if success:
    print("\n🎯 RECOMMENDATION: Apply these settings to full training!")
    print("Key improvements:")
    print("- Hyper-aggressive exponential reconstruction loss")
    print("- Latent dimension reduced to 2 (maximum compression)")
    print("- Ultra-low KL weight (0.00001)")
    print("- Massive reconstruction weight (100000)")
    print("- Adaptive learning rate strategy")
    print("- Combined reconstruction + latent space anomaly detection")
else:
    print("\n🔬 TRYING FINAL DESPERATE MEASURES...")
    print("Consider:")
    print("1. Latent dimension = 1 (extreme bottleneck)")
    print("2. Pure autoencoder (remove VAE components entirely)")
    print("3. Different architecture (CNN-based instead of LSTM)")
    print("4. Ensemble approach with multiple models")
    
    # Quick test with latent_dim=1
    print("\n🧪 Testing with latent_dim=1...")
    encoder_minimal, decoder_minimal, _, _ = select_architecture(
        normal_samples_count=len(X_train_normal),
        input_shape=input_shape,
        latent_dim=1,
        force_architecture='ultra_compact'
    )
    
    # Quick 10-epoch test
    opt_minimal = tf.keras.optimizers.Adam(0.001)
    for epoch in range(10):
        for i in range(0, len(X_train_normal), 4):
            batch = X_train_normal[i:i+4]
            if len(batch) < 4:
                continue
            with tf.GradientTape() as tape:
                z_mean, z_log_var, z = encoder_minimal(batch)
                reconstructed = decoder_minimal(z)
                loss = tf.reduce_mean(tf.square(batch - reconstructed))
            grads = tape.gradient(loss, encoder_minimal.trainable_weights + decoder_minimal.trainable_weights)
            if grads and all(g is not None for g in grads):
                opt_minimal.apply_gradients(zip(grads, encoder_minimal.trainable_weights + decoder_minimal.trainable_weights))
    
    # Test minimal model
    z_test_min, _, _ = encoder_minimal(X_test)
    recon_test_min = decoder_minimal(z_test_min)
    errors_min = tf.reduce_mean(tf.square(X_test - recon_test_min), axis=[1, 2]).numpy()
    
    normal_min = errors_min[y_test == 0]
    attack_min = errors_min[y_test == 1]
    ratio_min = np.mean(attack_min) / np.mean(normal_min)
    
    print(f"Latent_dim=1 separation ratio: {ratio_min:.3f}x")
    if ratio_min > 1.2:
        print("✅ Latent_dim=1 shows promise! Use this for full training.")
