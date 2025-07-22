"""
Training script for hyper-minimal VAE models to address severe overfitting
"""
import numpy as np
import tensorflow as tf
from models.hyper_minimal_vae import build_hyper_minimal_vae, build_extreme_minimal_vae

# Load data
sequence_path = "data/sequences"
X_train = np.load(f"{sequence_path}/X_train.npy")
X_test = np.load(f"{sequence_path}/X_test.npy")
y_train = np.load(f"{sequence_path}/y_train_binary.npy")
y_test = np.load(f"{sequence_path}/y_test_binary.npy")

# Get only normal training data
X_train_normal = X_train[y_train == 0]
print(f"📊 Training with {len(X_train_normal)} normal samples")
print(f"📊 Test data: {len(X_test)} samples ({np.sum(y_test)} FDIA)")

# Test different minimal architectures
architectures = [
    ("hyper_minimal", lambda: build_hyper_minimal_vae((12, 18), latent_dim=1)),
    ("extreme_minimal", lambda: build_extreme_minimal_vae((12, 18), latent_dim=1)),
]

def train_minimal_vae(encoder, decoder, X_train_normal, epochs=200):
    """Train minimal VAE with high reconstruction loss weight"""
    
    # Optimizers
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    # Loss functions  
    def kl_loss(z_mean, z_log_var):
        return -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    
    def reconstruction_loss(y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))
    
    # Training parameters
    kl_weight = 0.1  # Low KL weight
    recon_weight = 1000.0  # Very high reconstruction weight
    batch_size = 32
    
    print(f"🏋️ Training with KL weight: {kl_weight}, Recon weight: {recon_weight}")
    
    best_loss = float('inf')
    patience = 50
    wait = 0
    
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_kl = 0
        epoch_recon = 0
        
        # Shuffle data
        indices = np.random.permutation(len(X_train_normal))
        
        for i in range(0, len(X_train_normal), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_x = X_train_normal[batch_indices]
            
            with tf.GradientTape() as tape:
                z_mean, z_log_var, z = encoder(batch_x)
                reconstructed = decoder(z)
                
                kl_loss_val = kl_loss(z_mean, z_log_var)
                recon_loss_val = reconstruction_loss(batch_x, reconstructed)
                
                total_loss = kl_weight * kl_loss_val + recon_weight * recon_loss_val
            
            # Update weights
            trainable_weights = encoder.trainable_weights + decoder.trainable_weights
            grads = tape.gradient(total_loss, trainable_weights)
            optimizer.apply_gradients(zip(grads, trainable_weights))
            
            epoch_loss += total_loss.numpy()
            epoch_kl += kl_loss_val.numpy()
            epoch_recon += recon_loss_val.numpy()
        
        # Average losses
        num_batches = len(range(0, len(X_train_normal), batch_size))
        avg_loss = epoch_loss / num_batches
        avg_kl = epoch_kl / num_batches
        avg_recon = epoch_recon / num_batches
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            wait = 0
            # Save models
            encoder.save(f"outputs/checkpoints/minimal_encoder_best.h5")
            decoder.save(f"outputs/checkpoints/minimal_decoder_best.h5")
        else:
            wait += 1
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d}: Loss={avg_loss:.6f}, KL={avg_kl:.6f}, Recon={avg_recon:.6f}, Best={best_loss:.6f}")
        
        if wait >= patience:
            print(f"⏹️ Early stopping at epoch {epoch}")
            break
    
    return encoder, decoder

def evaluate_separation(encoder, decoder, X_test, y_test, X_train_normal):
    """Quick separation evaluation"""
    
    # Compute reconstruction errors
    def compute_errors(X_data):
        z_mean, z_log_var, z = encoder(X_data)
        reconstructed = decoder(z)
        errors = tf.reduce_mean(tf.square(X_data - reconstructed), axis=[1, 2])
        return errors.numpy()
    
    train_errors = compute_errors(X_train_normal)
    test_errors = compute_errors(X_test)
    
    normal_test_errors = test_errors[y_test == 0]
    fdia_test_errors = test_errors[y_test == 1]
    
    separation_ratio = np.mean(fdia_test_errors) / np.mean(normal_test_errors)
    
    print(f"🎯 SEPARATION ANALYSIS:")
    print(f"   Train Normal: {np.mean(train_errors):.6f} ± {np.std(train_errors):.6f}")
    print(f"   Test Normal:  {np.mean(normal_test_errors):.6f} ± {np.std(normal_test_errors):.6f}")
    print(f"   Test FDIA:    {np.mean(fdia_test_errors):.6f} ± {np.std(fdia_test_errors):.6f}")
    print(f"   Separation:   {separation_ratio:.3f}x {'✅' if separation_ratio > 2.0 else '❌'}")
    
    return separation_ratio

# Train and evaluate each architecture
results = {}

for name, build_func in architectures:
    print(f"\n{'='*60}")
    print(f"🧪 TESTING {name.upper()} ARCHITECTURE")
    print(f"{'='*60}")
    
    try:
        encoder, decoder, _ = build_func()
        
        # Train
        print(f"🏋️ Training {name}...")
        encoder, decoder = train_minimal_vae(encoder, decoder, X_train_normal, epochs=200)
        
        # Evaluate
        separation = evaluate_separation(encoder, decoder, X_test, y_test, X_train_normal)
        results[name] = separation
        
        print(f"✅ {name}: Separation = {separation:.3f}x")
        
    except Exception as e:
        print(f"❌ {name} failed: {e}")
        results[name] = 0.0

# Summary
print(f"\n{'='*60}")
print(f"📊 FINAL RESULTS SUMMARY")
print(f"{'='*60}")

for name, separation in results.items():
    status = "✅ GOOD" if separation > 2.0 else "⚠️ POOR" if separation > 1.5 else "❌ FAIL"
    print(f"{name:20}: {separation:.3f}x {status}")

best_arch = max(results.items(), key=lambda x: x[1])
print(f"\n🏆 BEST ARCHITECTURE: {best_arch[0]} ({best_arch[1]:.3f}x separation)")

if best_arch[1] > 2.0:
    print(f"🎉 SUCCESS! Use {best_arch[0]} for production")
else:
    print(f"⚠️ All architectures show poor separation")
    print(f"💡 Consider: Data quality issues, need for different approach")
