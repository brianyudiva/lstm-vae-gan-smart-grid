import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import os
import sys
from datetime import datetime
from sklearn.metrics import accuracy_score, average_precision_score

sys.path.append('/home/brianyudiva/Documents/Project/lstm-vae-gan-smart-grid')

from models.baseline_models import get_baseline_model
from utils.loss_functions import vae_loss, combined_loss, wasserstein_loss
from utils.utils import create_anomaly_labels


def evaluate_model_accuracy(model, X_test, y_test):
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
    
    thresholds = np.linspace(np.min(recon_errors), np.max(recon_errors), 100)
    best_accuracy = 0.0
    best_threshold = 0.0
    
    for threshold in thresholds:
        y_pred = (recon_errors > threshold).astype(int)
        accuracy = accuracy_score(y_test, y_pred)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    
    return best_accuracy, best_threshold, pr_auc


def load_best_hyperparameters():
    return {
        'latent_dim': 32,
        'lstm_units_1': 32,
        'lstm_units_2': 16,
        'dense_units_1': 32,
        'dense_units_2': 16,
        'dropout_rate': 0.3,
        'l1_reg': 5e-4,
        'l2_reg': 1e-3,
        'learning_rate': 0.0039659761145096435
    }


def load_data():
    data_path = '/home/brianyudiva/Documents/Project/lstm-vae-gan-smart-grid/data/sequences'
    
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
    
    best_accuracy = 0.0
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(epochs):
        epoch_losses = []
        
        for step in range(steps_per_epoch):
            batch_idx = np.random.randint(0, X_train.shape[0], batch_size)
            batch_x = X_train[batch_idx]
            
            loss = autoencoder.train_on_batch(batch_x, batch_x)
            epoch_losses.append(loss)
        
        avg_loss = np.mean(epoch_losses)
        
        if epoch % 5 == 0 or epoch == epochs - 1:
            accuracy, threshold, pr_auc = evaluate_model_accuracy(autoencoder, X_test, y_test)
            
            print(f"Epoch {epoch + 1:3d}: Loss: {avg_loss:.4f}, Accuracy: {accuracy:.3f}, PR-AUC: {pr_auc:.3f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_epoch = epoch
                patience_counter = 0

                autoencoder.save('/home/brianyudiva/Documents/Project/lstm-vae-gan-smart-grid/outputs/checkpoints/lstm_autoencoder_full.h5')
                encoder.save('/home/brianyudiva/Documents/Project/lstm-vae-gan-smart-grid/outputs/checkpoints/lstm_autoencoder_encoder.h5')
                decoder.save('/home/brianyudiva/Documents/Project/lstm-vae-gan-smart-grid/outputs/checkpoints/lstm_autoencoder_decoder.h5')
            else:
                patience_counter += 5
            
            if patience_counter >= 25:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        elif epoch % 1 == 0:
            print(f"Epoch {epoch + 1:3d}: Loss: {avg_loss:.4f}")
    
    print(f"Best accuracy: {best_accuracy:.3f} at epoch {best_epoch + 1}")
    
    class DummyHistory:
        def __init__(self):
            self.history = {'loss': epoch_losses}
    
    return DummyHistory(), autoencoder, best_accuracy


def train_vae_gan(X_train, X_test, y_test, params, epochs=50, batch_size=32):
    print("\n" + "="*60)
    print("TRAINING VAE-GAN")
    print("="*60)
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    model_params = {
        'latent_dim': params['latent_dim'],
        'dense_units': [128, 64, 32], 
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
    
    disc_optimizer = keras.optimizers.Adam(learning_rate=params['learning_rate'] * 0.5)
    discriminator.compile(optimizer=disc_optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    steps_per_epoch = min(X_train.shape[0] // batch_size, 80)
    print(f"Training config: Epochs: {epochs}, Steps per epoch: {steps_per_epoch}, Batch size: {batch_size}")
    
    vae_losses = []
    disc_losses = []
    best_accuracy = 0.0
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(epochs):
        epoch_vae_losses = []
        epoch_disc_losses = []
        
        for step in range(steps_per_epoch):
            batch_idx = np.random.randint(0, X_train.shape[0], batch_size)
            batch_x = X_train[batch_idx]
            
            vae_loss_val = vae.train_on_batch(batch_x, batch_x)
            epoch_vae_losses.append(vae_loss_val)
            
            z_mean, z_log_var, z = encoder.predict(batch_x, verbose=0)
            fake_data = decoder.predict(z, verbose=0)
            
            d_loss_real = discriminator.train_on_batch(batch_x, np.ones((len(batch_x), 1)))
            d_loss_fake = discriminator.train_on_batch(fake_data, np.zeros((len(fake_data), 1)))
            d_loss = 0.5 * (d_loss_real[0] + d_loss_fake[0])
            epoch_disc_losses.append(d_loss)
        
        avg_vae_loss = np.mean(epoch_vae_losses)
        avg_disc_loss = np.mean(epoch_disc_losses)
        vae_losses.append(avg_vae_loss)
        disc_losses.append(avg_disc_loss)
        
        if epoch % 5 == 0 or epoch == epochs - 1:
            accuracy, threshold, pr_auc = evaluate_model_accuracy(vae, X_test, y_test)
            print(f"Epoch {epoch + 1:3d}: Accuracy: {accuracy:.3f}, PR-AUC: {pr_auc:.3f}, VAE Loss: {avg_vae_loss:.4f}, Disc Loss: {avg_disc_loss:.4f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_epoch = epoch
                patience_counter = 0

                encoder.save('/home/brianyudiva/Documents/Project/lstm-vae-gan-smart-grid/outputs/checkpoints/vae_gan_encoder.h5')
                decoder.save('/home/brianyudiva/Documents/Project/lstm-vae-gan-smart-grid/outputs/checkpoints/vae_gan_decoder.h5')
                discriminator.save('/home/brianyudiva/Documents/Project/lstm-vae-gan-smart-grid/outputs/checkpoints/vae_gan_discriminator.h5')
                vae.save('/home/brianyudiva/Documents/Project/lstm-vae-gan-smart-grid/outputs/checkpoints/vae_gan_full.h5')
            else:
                patience_counter += 5
            
            if patience_counter >= 25:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        elif epoch % 1 == 0:
            print(f"Epoch {epoch + 1:3d}: VAE Loss: {avg_vae_loss:.4f}, Disc Loss: {avg_disc_loss:.4f}")
    
    print(f"Best accuracy: {best_accuracy:.3f} at epoch {best_epoch + 1}")
    
    return {'vae_losses': vae_losses, 'disc_losses': disc_losses}, vae, best_accuracy


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
    best_accuracy = 0.0
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(epochs):
        epoch_gen_losses = []
        epoch_disc_losses = []
        
        for step in range(steps_per_epoch):
            batch_idx = np.random.randint(0, X_train.shape[0], batch_size)
            batch_x = X_train[batch_idx]
            
            gen_loss = generator.train_on_batch(batch_x, batch_x)
            epoch_gen_losses.append(gen_loss)
            
            fake_data = generator.predict(batch_x, verbose=0)
            
            d_loss_real = discriminator.train_on_batch(batch_x, np.ones((len(batch_x), 1)))
            d_loss_fake = discriminator.train_on_batch(fake_data, np.zeros((len(fake_data), 1)))
            d_loss = 0.5 * (d_loss_real[0] + d_loss_fake[0])
            epoch_disc_losses.append(d_loss)
        
        avg_gen_loss = np.mean(epoch_gen_losses)
        avg_disc_loss = np.mean(epoch_disc_losses)
        gen_losses.append(avg_gen_loss)
        disc_losses.append(avg_disc_loss)
        
        if epoch % 5 == 0 or epoch == epochs - 1:
            accuracy, threshold, pr_auc = evaluate_model_accuracy(generator, X_test, y_test)
            print(f"Epoch {epoch + 1:3d}: Accuracy: {accuracy:.3f}, PR-AUC: {pr_auc:.3f}, Gen Loss: {avg_gen_loss:.4f}, Disc Loss: {avg_disc_loss:.4f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_epoch = epoch
                patience_counter = 0

                encoder.save('/home/brianyudiva/Documents/Project/lstm-vae-gan-smart-grid/outputs/checkpoints/lstm_gan_encoder.h5')
                decoder.save('/home/brianyudiva/Documents/Project/lstm-vae-gan-smart-grid/outputs/checkpoints/lstm_gan_decoder.h5')
                discriminator.save('/home/brianyudiva/Documents/Project/lstm-vae-gan-smart-grid/outputs/checkpoints/lstm_gan_discriminator.h5')
                generator.save('/home/brianyudiva/Documents/Project/lstm-vae-gan-smart-grid/outputs/checkpoints/lstm_gan_generator.h5')
            else:
                patience_counter += 5
            
            if patience_counter >= 25:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        elif epoch % 1 == 0:
            print(f"Epoch {epoch + 1:3d}: Gen Loss: {avg_gen_loss:.4f}, Disc Loss: {avg_disc_loss:.4f}")
    
    print(f"Best accuracy: {best_accuracy:.3f} at epoch {best_epoch + 1}")
    
    return {'gen_losses': gen_losses, 'disc_losses': disc_losses}, generator, best_accuracy


def train_lstm_vae(X_train, X_test, y_test, params, epochs=50, batch_size=32):
    print("\n" + "="*60)
    print("TRAINING LSTM-VAE")
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
    
    encoder, decoder, vae = get_baseline_model('lstm_vae', input_shape, **model_params)
    
    optimizer = keras.optimizers.Adam(learning_rate=params['learning_rate'])
    vae.compile(
        optimizer=optimizer,
        loss=lambda y_true, y_pred: vae_loss(y_true, y_pred, encoder, params['latent_dim'], beta=1.0)
    )
    
    steps_per_epoch = min(X_train.shape[0] // batch_size, 100)
    print(f"Training config: Epochs: {epochs}, Steps per epoch: {steps_per_epoch}, Batch size: {batch_size}")
    
    best_accuracy = 0.0
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(epochs):
        epoch_losses = []
        
        for step in range(steps_per_epoch):
            batch_idx = np.random.randint(0, X_train.shape[0], batch_size)
            batch_x = X_train[batch_idx]
            
            loss = vae.train_on_batch(batch_x, batch_x)
            epoch_losses.append(loss)
        
        avg_loss = np.mean(epoch_losses)
        
        if epoch % 5 == 0 or epoch == epochs - 1:
            accuracy, threshold, pr_auc = evaluate_model_accuracy(vae, X_test, y_test)
            
            print(f"Epoch {epoch + 1:3d}: Loss: {avg_loss:.4f}, Accuracy: {accuracy:.3f}, PR-AUC: {pr_auc:.3f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_epoch = epoch
                patience_counter = 0

                encoder.save('/home/brianyudiva/Documents/Project/lstm-vae-gan-smart-grid/outputs/checkpoints/lstm_vae_encoder.h5')
                decoder.save('/home/brianyudiva/Documents/Project/lstm-vae-gan-smart-grid/outputs/checkpoints/lstm_vae_decoder.h5')
                vae.save('/home/brianyudiva/Documents/Project/lstm-vae-gan-smart-grid/outputs/checkpoints/lstm_vae_full.h5')
            else:
                patience_counter += 5
            
            if patience_counter >= 25:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        elif epoch % 1 == 0:
            print(f"Epoch {epoch + 1:3d}: Loss: {avg_loss:.4f}")
    
    print(f"Best accuracy: {best_accuracy:.3f} at epoch {best_epoch + 1}")
    
    class DummyHistory:
        def __init__(self):
            self.history = {'loss': epoch_losses}
    
    return DummyHistory(), vae, best_accuracy


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
        ('lstm_vae', train_lstm_vae)
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
    
    summary_path = '/home/brianyudiva/Documents/Project/lstm-vae-gan-smart-grid/outputs/checkpoints/baseline_training_summary.json'
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
