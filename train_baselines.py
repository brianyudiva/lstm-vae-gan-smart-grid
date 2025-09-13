import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import os
from datetime import datetime
from sklearn.metrics import average_precision_score
from models.baseline_models import get_baseline_model
from utils.loss_functions import beta_schedule, kl_loss, regularization_loss, robust_reconstruction_loss, vae_loss


def evaluate_model(model, X_test, y_test):
    if hasattr(model, 'predict'):
        predictions = model.predict(X_test, verbose=0)
    else:
        encoder, decoder = model
        latent = encoder.predict(X_test, verbose=0)
        if isinstance(latent, list):
            latent = latent[2]
        predictions = decoder.predict(latent, verbose=0)
    
    recon_errors = np.mean(np.square(X_test - predictions), axis=(1, 2))
    
    pr_auc = average_precision_score(y_test, recon_errors)

    normal_recon_error = np.mean(recon_errors[y_test == 0])
    
    return pr_auc, normal_recon_error


def load_best_hyperparameters():
    return {
        'latent_dim': 16,
        'lstm_units_1': 16,
        'lstm_units_2': 8,
        'dense_units_1': 16,
        'dense_units_2': 8,
        'dropout_rate': 0.3,
        'l1_reg': 5e-4,
        'l2_reg': 1e-3,
        'learning_rate': 0.0039659761145096435
    }


def load_data():
    data_path = 'data/sequences'
    
    X_train = np.load(os.path.join(data_path, 'X_train.npy'))
    X_test = np.load(os.path.join(data_path, 'X_test.npy'))
    y_train = np.load(os.path.join(data_path, 'y_train_binary.npy'))
    y_test = np.load(os.path.join(data_path, 'y_test_binary.npy'))
    
    print(f"Data loaded:")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_test shape: {X_test.shape}")
    print(f"  y_train shape: {y_train.shape}")
    print(f"  y_test shape: {y_test.shape}")
    
    return X_train, X_test, y_train, y_test


def train_lstm_autoencoder(X_train, X_test, y_test, params, epochs=50, batch_size=32):
    print("\n" + "="*60)
    print("TRAINING LSTM AUTOENCODER")
    print("="*60)
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    model_params = {
        'latent_dim': params['latent_dim'],
        'lstm_units_1': params['lstm_units_1'],
        'lstm_units_2': params['lstm_units_2'], 
        'dense_units_1': params['dense_units_1'],
        'dense_units_2': params['dense_units_2'],
        'dropout_rate': params['dropout_rate'],
        'l1_reg': params['l1_reg'],
        'l2_reg': params['l2_reg']
    }
    
    encoder, decoder, autoencoder = get_baseline_model('lstm_autoencoder', input_shape, **model_params)
    
    optimizer = keras.optimizers.Adam(learning_rate=params['learning_rate'])
    autoencoder.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    steps_per_epoch = min(X_train.shape[0] // batch_size, 100)
    print(f"Training config: Epochs: {epochs}, Steps per epoch: {steps_per_epoch}, Batch size: {batch_size}")
    
    best_pr_auc = 0.0
    best_recon_error = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(epochs):
        epoch_losses = []
        
        for _ in range(steps_per_epoch):
            batch_idx = np.random.randint(0, X_train.shape[0], batch_size)
            batch_x = X_train[batch_idx]
            
            loss = autoencoder.train_on_batch(batch_x, batch_x)
            epoch_losses.append(loss)
        
        avg_loss = np.mean(epoch_losses)
        
        if epoch % 5 == 0 or epoch == epochs - 1:
            pr_auc, recon_error = evaluate_model(autoencoder, X_test, y_test)
            
            print(f"Epoch {epoch + 1:3d}: Loss: {avg_loss:.4f}, PR-AUC: {pr_auc:.3f}")
            
            if pr_auc > best_pr_auc or recon_error < best_recon_error:
                best_pr_auc = pr_auc
                best_epoch = epoch
                best_recon_error = recon_error
                patience_counter = 0

                autoencoder.save('/outputs/checkpoints/lstm_autoencoder_full.h5')
                encoder.save('/outputs/checkpoints/lstm_autoencoder_encoder.h5')
                decoder.save('/outputs/checkpoints/lstm_autoencoder_decoder.h5')
            else:
                patience_counter += 5
            
            if patience_counter >= 15:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        elif epoch % 1 == 0:
            print(f"Epoch {epoch + 1:3d}: Loss: {avg_loss:.4f}")
    
    print(f"Best PR-AUC: {best_pr_auc:.3f} at epoch {best_epoch + 1}")
    
    class DummyHistory:
        def __init__(self):
            self.history = {'loss': epoch_losses}
    
    return DummyHistory(), autoencoder, best_pr_auc


def train_vae_gan(X_train, X_test, y_test, params, epochs=50, batch_size=32):
    print("\n" + "="*60)
    print("TRAINING VAE-GAN")
    print("="*60)
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    model_params = {
        'latent_dim': params['latent_dim'],
        'dense_units': [32, 16], 
        'dropout_rate': params['dropout_rate'],
        'l1_reg': params['l1_reg'],
        'l2_reg': params['l2_reg']
    }
    
    encoder, decoder, discriminator, vae = get_baseline_model('vae_gan', input_shape, **model_params)
    vae_optimizer = keras.optimizers.Adam(learning_rate=params['learning_rate'])
    vae.compile(
        optimizer=vae_optimizer,
        loss=lambda y_true, y_pred: vae_loss(y_true, y_pred, encoder, params['latent_dim'], beta=1.0)
    )
    
    gen_optimizer = keras.optimizers.Adam(learning_rate=params['learning_rate'])
    decoder.compile(optimizer=gen_optimizer, loss='mse')
    
    disc_optimizer = keras.optimizers.Adam(learning_rate=params['learning_rate'] * 0.5)
    discriminator.compile(optimizer=disc_optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    steps_per_epoch = min(X_train.shape[0] // batch_size, 80)
    print(f"Training config: Epochs: {epochs}, Steps per epoch: {steps_per_epoch}, Batch size: {batch_size}")
    
    vae_losses = []
    disc_losses = []
    best_pr_auc = 0.0
    best_recon_error = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    kl_weight = 1.0
    beta_warmup_epochs = epochs // 4
    reconstruction_weight = 1.0
    regularization_weight = 0.01
    
    for epoch in range(epochs):
        epoch_vae_losses = []
        epoch_disc_losses = []
        
        for _ in range(steps_per_epoch):
            batch_idx = np.random.randint(0, X_train.shape[0], batch_size)
            batch_x = X_train[batch_idx]

            # === Train Discriminator ===
            with tf.GradientTape() as disc_tape:
                real_pred = discriminator(batch_x, training=True)
                real_labels = tf.ones_like(real_pred)
                real_loss = tf.keras.losses.binary_crossentropy(real_labels, real_pred)
                
                z_mean, z_log_var, z = encoder(batch_x, training=False)
                fake_batch = decoder(z, training=False)
                fake_pred = discriminator(fake_batch, training=True)
                fake_labels = tf.zeros_like(fake_pred)
                fake_loss = tf.keras.losses.binary_crossentropy(fake_labels, fake_pred)
                
                disc_loss = tf.reduce_mean(real_loss + fake_loss)
            
            disc_grads = disc_tape.gradient(disc_loss, discriminator.trainable_weights)
            if disc_grads:
                disc_grads = [tf.clip_by_norm(g, 1.0) for g in disc_grads]
                disc_optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_weights))
            
            # === Train Generator (VAE) ===
            with tf.GradientTape() as gen_tape:
                z_mean, z_log_var, z = encoder(batch_x, training=True)
                reconstructed = decoder(z, training=True)
                
                recon_loss_val = robust_reconstruction_loss(batch_x, reconstructed)
                
                kl_beta = beta_schedule(epoch, epochs, 
                                        max_beta=kl_weight, 
                                        warmup_epochs=beta_warmup_epochs)
                kl_loss_val = kl_beta * kl_loss(z_mean, z_log_var)
                
                reg_loss_val = regularization_loss(encoder, decoder)
                
                gen_pred = discriminator(reconstructed, training=False)
                gen_labels = tf.ones_like(gen_pred)
                adversarial_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(gen_labels, gen_pred))
                
                gen_loss = (reconstruction_weight * recon_loss_val +
                            kl_loss_val +
                            regularization_weight * reg_loss_val +
                            0.1 * adversarial_loss)
                
            gen_grads = gen_tape.gradient(gen_loss, encoder.trainable_weights + decoder.trainable_weights)
            if gen_grads:
                gen_grads = [tf.clip_by_norm(g, 1.0) for g in gen_grads]
                gen_optimizer.apply_gradients(zip(gen_grads, encoder.trainable_weights + decoder.trainable_weights))
            
            epoch_vae_losses.append(float(gen_loss.numpy()))
            epoch_disc_losses.append(float(disc_loss.numpy()))
        
        avg_vae_loss = np.mean(epoch_vae_losses)
        avg_disc_loss = np.mean(epoch_disc_losses)
        vae_losses.append(avg_vae_loss)
        disc_losses.append(avg_disc_loss)
        
        if epoch % 5 == 0 or epoch == epochs - 1:
            pr_auc, recon_error = evaluate_model(vae, X_test, y_test)
            print(f"Epoch {epoch + 1:3d}: PR-AUC: {pr_auc:.3f}, VAE Loss: {avg_vae_loss:.4f}, Disc Loss: {avg_disc_loss:.4f}")
            
            if pr_auc > best_pr_auc or recon_error < best_recon_error:
                best_pr_auc = pr_auc
                best_recon_error = recon_error
                best_epoch = epoch
                patience_counter = 0

                encoder.save('/outputs/checkpoints/vae_gan_encoder.h5')
                decoder.save('/outputs/checkpoints/vae_gan_decoder.h5')
                discriminator.save('/outputs/checkpoints/vae_gan_discriminator.h5')
                vae.save('/outputs/checkpoints/vae_gan_full.h5')
                print(f"Model saved! PR-AUC: {pr_auc:.4f}, Recon Error: {recon_error:.6f}")
            else:
                patience_counter += 5
            
            if patience_counter >= 15:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        elif epoch % 1 == 0:
            print(f"Epoch {epoch + 1:3d}: VAE Loss: {avg_vae_loss:.4f}, Disc Loss: {avg_disc_loss:.4f}")
    
    print(f"Best PR-AUC: {best_pr_auc:.3f} at epoch {best_epoch + 1}")
    
    return {'vae_losses': vae_losses, 'disc_losses': disc_losses}, vae, best_pr_auc


def train_lstm_gan(X_train, X_test, y_test, params, epochs=50, batch_size=32):
    print("\n" + "="*60)
    print("TRAINING LSTM-GAN")
    print("="*60)
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    model_params = {
        'latent_dim': params['latent_dim'],
        'lstm_units_1': params['lstm_units_1'],
        'lstm_units_2': params['lstm_units_2'],
        'dense_units_1': params['dense_units_1'],
        'dense_units_2': params['dense_units_2'],
        'dropout_rate': params['dropout_rate'],
        'l1_reg': params['l1_reg'],
        'l2_reg': params['l2_reg']
    }
    
    encoder, decoder, discriminator, generator = get_baseline_model('lstm_gan', input_shape, **model_params)
    
    gen_optimizer = keras.optimizers.Adam(learning_rate=params['learning_rate'])
    generator.compile(optimizer=gen_optimizer, loss='mse')
    
    disc_optimizer = keras.optimizers.Adam(learning_rate=params['learning_rate'] * 0.5)
    discriminator.compile(optimizer=disc_optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    steps_per_epoch = min(X_train.shape[0] // batch_size, 80)
    print(f"Training config: Epochs: {epochs}, Steps per epoch: {steps_per_epoch}, Batch size: {batch_size}")
    
    gen_losses = []
    disc_losses = []
    best_pr_auc = 0.0
    best_recon_error = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(epochs):
        epoch_gen_losses = []
        epoch_disc_losses = []
        
        for step in range(steps_per_epoch):
            batch_idx = np.random.randint(0, X_train.shape[0], batch_size)
            batch_x = X_train[batch_idx]
            
            # === Train Generator (Autoencoder) ===
            with tf.GradientTape() as gen_tape:
                latent = encoder(batch_x, training=True)
                if isinstance(latent, list):
                    latent = latent[2] if len(latent) > 2 else latent[0]
                
                reconstructed = decoder(latent, training=True)
                
                gen_loss = tf.reduce_mean(tf.keras.losses.mse(batch_x, reconstructed))
                
                reg_loss = regularization_loss(encoder, decoder)
                gen_loss += 0.01 * reg_loss
            
            gen_grads = gen_tape.gradient(gen_loss, encoder.trainable_weights + decoder.trainable_weights)
            if gen_grads:
                gen_grads = [tf.clip_by_norm(g, 1.0) for g in gen_grads]
                gen_optimizer.apply_gradients(zip(gen_grads, encoder.trainable_weights + decoder.trainable_weights))
            
            # === Train Discriminator ===
            with tf.GradientTape() as disc_tape:
                real_pred = discriminator(batch_x, training=True)
                real_labels = tf.ones_like(real_pred)
                real_loss = tf.keras.losses.binary_crossentropy(real_labels, real_pred)
                
                latent = encoder(batch_x, training=False)
                if isinstance(latent, list):
                    latent = latent[2] if len(latent) > 2 else latent[0]
                fake_batch = decoder(latent, training=False)
                fake_pred = discriminator(fake_batch, training=True)
                fake_labels = tf.zeros_like(fake_pred)
                fake_loss = tf.keras.losses.binary_crossentropy(fake_labels, fake_pred)
                
                disc_loss = tf.reduce_mean(real_loss + fake_loss)
            
            disc_grads = disc_tape.gradient(disc_loss, discriminator.trainable_weights)
            if disc_grads:
                disc_grads = [tf.clip_by_norm(g, 1.0) for g in disc_grads]
                disc_optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_weights))
            
            epoch_gen_losses.append(float(gen_loss.numpy()))
            epoch_disc_losses.append(float(disc_loss.numpy()))
        
        avg_gen_loss = np.mean(epoch_gen_losses)
        avg_disc_loss = np.mean(epoch_disc_losses)
        gen_losses.append(avg_gen_loss)
        disc_losses.append(avg_disc_loss)
        
        if epoch % 5 == 0 or epoch == epochs - 1:
            pr_auc, recon_error = evaluate_model(generator, X_test, y_test)
            print(f"Epoch {epoch + 1:3d}: PR-AUC: {pr_auc:.3f}, Gen Loss: {avg_gen_loss:.4f}, Disc Loss: {avg_disc_loss:.4f}")
            
            if pr_auc > best_pr_auc or recon_error < best_recon_error:
                best_pr_auc = pr_auc
                best_recon_error = recon_error
                best_epoch = epoch
                patience_counter = 0

                encoder.save('/outputs/checkpoints/lstm_gan_encoder.h5')
                decoder.save('/outputs/checkpoints/lstm_gan_decoder.h5')
                discriminator.save('/outputs/checkpoints/lstm_gan_discriminator.h5')
                generator.save('/outputs/checkpoints/lstm_gan_generator.h5')
                print(f"  Model saved! PR-AUC: {pr_auc:.4f}, Recon Error: {recon_error:.6f}")
            else:
                patience_counter += 5
            
            if patience_counter >= 25:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        elif epoch % 1 == 0:
            print(f"Epoch {epoch + 1:3d}: Gen Loss: {avg_gen_loss:.4f}, Disc Loss: {avg_disc_loss:.4f}")
    
    print(f"Best PR-AUC: {best_pr_auc:.3f} at epoch {best_epoch + 1}")
    
    return {'gen_losses': gen_losses, 'disc_losses': disc_losses}, generator, best_pr_auc

def main():    
    print("Starting baseline model training...")
    
    X_train, X_test, y_train, y_test = load_data()
    params = load_best_hyperparameters()
    
    epochs = 50
    batch_size = 32
    
    training_results = {}
    
    models_to_train = [
        ('lstm_autoencoder', train_lstm_autoencoder),
        ('vae_gan', train_vae_gan), 
        ('lstm_gan', train_lstm_gan),
    ]
    
    for model_name, train_func in models_to_train:
        try:
            print(f"\n{'='*80}")
            print(f"Starting training for: {model_name.upper()}")
            print(f"{'='*80}")
            
            start_time = datetime.now()
            history, model, best_accuracy = train_func(X_train, X_test, y_test, params, epochs, batch_size)
            end_time = datetime.now()
            
            training_time = (end_time - start_time).total_seconds()
            
            training_results[model_name] = {
                'training_time_seconds': training_time,
                'history': history if hasattr(history, 'history') else history,
                'model_params': model.count_params(),
                'best_accuracy': float(best_accuracy),
                'completed_at': end_time.isoformat()
            }
            
            print(f"{model_name} training completed in {training_time:.2f} seconds")
            print(f"Best accuracy: {best_accuracy:.3f}")
            
        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
            training_results[model_name] = {
                'error': str(e),
                'completed_at': datetime.now().isoformat()
            }
    
    summary_path = 'outputs/checkpoints/baseline_training_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(training_results, f, indent=2, default=str)
    
    print(f"\n{'='*80}")
    print("BASELINE MODEL TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Training summary saved to: {summary_path}")
    
    for model_name, results in training_results.items():
        if 'error' not in results:
            accuracy_str = f", Accuracy: {results['best_accuracy']:.3f}" if 'best_accuracy' in results else ""
            print(f"✅ {model_name}: {results['model_params']:,} params, {results['training_time_seconds']:.1f}s{accuracy_str}")
        else:
            print(f"❌ {model_name}: {results['error']}")


if __name__ == "__main__":
    main()
