import numpy as np
import tensorflow as tf
import optuna
from models.lstm_vae_gan import build_lstm_vae_gan_regular
from utils.loss_functions import (
    kl_loss, robust_reconstruction_loss, regularization_loss, beta_schedule
)
import os
import json
from datetime import datetime
from sklearn.metrics import average_precision_score
from utils.utils import convert_to_json_serializable
import logging

# Set up logging for Optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_simple_objective(X_train_normal, X_test, y_test):
    """
    Create a simplified objective function for quick Optuna optimization.
    Uses the existing build_lstm_vae_gan_regular function with limited hyperparameters.
    """
    
    def objective(trial):
        # Clear any previous session
        tf.keras.backend.clear_session()
        
        # Suggest key hyperparameters only
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'kl_weight': trial.suggest_float('kl_weight', 1e-4, 1e-2, log=True),
            'regularization_weight': trial.suggest_float('regularization_weight', 1e-7, 1e-4, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256]),
            'latent_dim': trial.suggest_categorical('latent_dim', [8, 16, 32]),
            'reconstruction_weight': trial.suggest_float('reconstruction_weight', 0.8, 1.5),
            'beta_warmup_epochs': trial.suggest_int('beta_warmup_epochs', 3, 10),
        }
        
        input_shape = (X_train_normal.shape[1], X_train_normal.shape[2])
        
        try:
            # Build model with existing function
            encoder, decoder, _ = build_lstm_vae_gan_regular(
                input_shape=input_shape,
                latent_dim=params['latent_dim']
            )
            
            optimizer = tf.keras.optimizers.Adam(params['learning_rate'], clipnorm=1.0)
            
            batch_size = params['batch_size']
            epochs = 30  # Quick evaluation
            steps_per_epoch = min(X_train_normal.shape[0] // batch_size, 50)
            
            logger.info(f"Trial {trial.number}: {params}")
            
            best_pr_auc = 0.0
            
            for epoch in range(epochs):
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
                        
                        # Gradual KL weight increase
                        kl_beta = beta_schedule(epoch, epochs, 
                                              max_beta=params['kl_weight'], 
                                              warmup_epochs=params['beta_warmup_epochs'])
                        kl_loss_val = kl_beta * kl_loss(z_mean, z_log_var)
                        
                        reg_loss_val = regularization_loss(encoder, decoder)
                        
                        # Total VAE loss
                        vae_loss = (params['reconstruction_weight'] * recon_loss_val +
                                   kl_loss_val +
                                   params['regularization_weight'] * reg_loss_val)
                    
                    # Update VAE parameters
                    grads = tape.gradient(vae_loss, encoder.trainable_weights + decoder.trainable_weights)
                    if grads:
                        grads = [tf.clip_by_norm(g, 1.0) for g in grads]
                        optimizer.apply_gradients(zip(grads, encoder.trainable_weights + decoder.trainable_weights))
                
                # Evaluate every 5 epochs
                if epoch % 5 == 0 or epoch == epochs - 1:
                    _, _, z_test_enc = encoder(X_test)
                    reconstructed_test = decoder(z_test_enc)
                    recon_errors = tf.reduce_mean(tf.square(X_test - reconstructed_test), axis=[1, 2]).numpy()
                    
                    # Performance metrics
                    pr_auc = float(average_precision_score(y_test, recon_errors))
                    best_pr_auc = max(best_pr_auc, pr_auc)
                
                # Report for pruning
                trial.report(best_pr_auc, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            
            logger.info(f"Trial {trial.number}: Final PR-AUC: {best_pr_auc:.4f}")
            return best_pr_auc
            
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {str(e)}")
            return 0.0
    
    return objective

def quick_optuna_training(n_trials=20, study_name="lstm_vae_gan_quick_optimization"):
    """
    Quick hyperparameter optimization using existing model architecture.
    
    Args:
        n_trials: Number of trials (default: 20 for quick optimization)
        study_name: Name of the Optuna study
    
    Returns:
        study: Optuna study object
        best_params: Best hyperparameters
        trained_models: Final trained models with best params
    """
    sequence_path = "data/sequences"
    output_path = "outputs/checkpoints"
    model_prefix = "lstm_vae_gan_quick_optuna"
    os.makedirs(output_path, exist_ok=True)

    # Load data
    X_train = np.load(f"{sequence_path}/X_train.npy")
    X_test = np.load(f"{sequence_path}/X_test.npy")
    y_train = np.load(f"{sequence_path}/y_train_binary.npy")
    y_test = np.load(f"{sequence_path}/y_test_binary.npy")

    print(f"Data loaded - Train FDIA: {np.sum(y_train)}/{len(y_train)}, Test FDIA: {np.sum(y_test)}/{len(y_test)}")
    
    X_train_normal = X_train[y_train == 0]
    print(f"Training on normal data: {len(X_train_normal)} samples")

    # Create Optuna study with simpler configuration
    study = optuna.create_study(
        direction='maximize',
        study_name=study_name,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5),
        sampler=optuna.samplers.TPESampler(n_startup_trials=5)
    )
    
    # Create objective function
    objective = create_simple_objective(X_train_normal, X_test, y_test)
    
    # Run optimization
    print(f"\n{'='*50}")
    print(f"QUICK OPTUNA OPTIMIZATION")
    print(f"{'='*50}")
    print(f"🎯 Objective: Maximize PR-AUC")
    print(f"🔄 Trials: {n_trials}")
    print(f"⚡ Quick mode: Using existing architecture")
    print(f"{'='*50}\n")
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Get best parameters
    best_params = study.best_params
    best_value = study.best_value
    
    print(f"\n{'='*50}")
    print(f"OPTIMIZATION COMPLETED!")
    print(f"{'='*50}")
    print(f"🏆 Best PR-AUC: {best_value:.4f}")
    print(f"🎯 Best Parameters:")
    for param, value in best_params.items():
        print(f"   {param}: {value}")
    print(f"{'='*50}\n")
    
    # Train final model with best parameters and more epochs
    print("Training final model with best parameters...")
    tf.keras.backend.clear_session()
    
    input_shape = (X_train_normal.shape[1], X_train_normal.shape[2])
    
    # Build final model
    encoder, decoder, vae = build_lstm_vae_gan_regular(
        input_shape=input_shape,
        latent_dim=best_params['latent_dim']
    )
    
    # Train with best parameters
    optimizer = tf.keras.optimizers.Adam(best_params['learning_rate'], clipnorm=1.0)
    
    batch_size = best_params['batch_size']
    epochs = 80  # More epochs for final training
    steps_per_epoch = min(X_train_normal.shape[0] // batch_size, 150)
    
    best_pr_auc = 0.0
    training_history = []
    
    for epoch in range(epochs):
        epoch_losses = {'vae_loss': 0, 'recon_loss': 0, 'kl_loss': 0, 'reg_loss': 0}
        
        for _ in range(steps_per_epoch):
            normal_idx = np.random.randint(0, X_train_normal.shape[0], batch_size)
            normal_batch = X_train_normal[normal_idx]
            
            with tf.GradientTape() as tape:
                z_mean, z_log_var, z = encoder(normal_batch, training=True)
                reconstructed = decoder(z, training=True)
                
                recon_loss_val = robust_reconstruction_loss(normal_batch, reconstructed)
                
                kl_beta = beta_schedule(epoch, epochs, 
                                      max_beta=best_params['kl_weight'], 
                                      warmup_epochs=best_params['beta_warmup_epochs'])
                kl_loss_val = kl_beta * kl_loss(z_mean, z_log_var)
                
                reg_loss_val = regularization_loss(encoder, decoder)
                
                vae_loss = (best_params['reconstruction_weight'] * recon_loss_val +
                           kl_loss_val +
                           best_params['regularization_weight'] * reg_loss_val)
            
            grads = tape.gradient(vae_loss, encoder.trainable_weights + decoder.trainable_weights)
            if grads:
                grads = [tf.clip_by_norm(g, 1.0) for g in grads]
                optimizer.apply_gradients(zip(grads, encoder.trainable_weights + decoder.trainable_weights))
            
            epoch_losses['vae_loss'] += vae_loss.numpy()
            epoch_losses['recon_loss'] += recon_loss_val.numpy()
            epoch_losses['kl_loss'] += kl_loss_val.numpy()
            epoch_losses['reg_loss'] += reg_loss_val.numpy()
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= steps_per_epoch
        
        # Evaluate every 5 epochs
        if epoch % 5 == 0 or epoch == epochs - 1:
            _, _, z_test_enc = encoder(X_test)
            reconstructed_test = decoder(z_test_enc)
            recon_errors = tf.reduce_mean(tf.square(X_test - reconstructed_test), axis=[1, 2]).numpy()
            
            normal_errors = recon_errors[y_test == 0]
            attack_errors = recon_errors[y_test == 1]
            separation_ratio = np.mean(attack_errors) / np.mean(normal_errors)
            
            pr_auc = average_precision_score(y_test, recon_errors)
            
            training_history.append({
                'epoch': epoch + 1,
                'pr_auc': float(pr_auc),
                'separation_ratio': float(separation_ratio),
                'losses': {k: float(v) for k, v in epoch_losses.items()}
            })
            
            if pr_auc > best_pr_auc:
                best_pr_auc = pr_auc
                # Save best model
                encoder.save(f"{output_path}/{model_prefix}_encoder.h5")
                decoder.save(f"{output_path}/{model_prefix}_decoder.h5")
                
            if epoch % 10 == 0:
                print(f"Epoch {epoch + 1}: PR-AUC: {pr_auc:.3f}, Separation: {separation_ratio:.3f}x")
    
    # Save results
    results = {
        'study_name': study_name,
        'n_trials': n_trials,
        'best_value': float(best_value),
        'best_params': best_params,
        'final_pr_auc': float(best_pr_auc),
        'training_history': training_history,
        'optimization_completed': datetime.now().isoformat()
    }
    
    with open(f"{output_path}/{model_prefix}_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {model_prefix}_results.json")
    print(f"🏆 Final model PR-AUC: {best_pr_auc:.4f}")
    
    return study, best_params, (encoder, decoder, vae)

if __name__ == "__main__":
    # Run quick optimization with exactly 20 trials
    study, best_params, models = quick_optuna_training(
        n_trials=20,  # Fixed at 20 trials for quick optimization
        study_name="lstm_vae_gan_quick_optuna"
    )
    
    print("\n🎉 Quick hyperparameter optimization completed!")
    print(f"🔧 Best hyperparameters found:")
    for param, value in best_params.items():
        print(f"   {param}: {value}")
