import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import json
import os
import sys
from sklearn.metrics import precision_recall_curve, recall_score, precision_score, f1_score
import seaborn as sns

# Add project root to path
sys.path.append('.')
from models.baseline_models import SamplingLayer

def set_reproducible_seeds(seed=42):
    """Set seeds for reproducible results"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # For older TensorFlow versions
    try:
        tf.compat.v1.set_random_seed(seed)
    except:
        pass

def load_model_safe(model_path, model_name=""):
    """Safely load a model with error handling and custom objects"""
    try:
        if os.path.exists(model_path):
            custom_objects = {
                'SamplingLayer': SamplingLayer,
                'sampling': SamplingLayer,
            }
            
            try:
                model = keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
                print(f"✅ Loaded {model_name}: {model_path}")
                return model
            except Exception as e:
                try:
                    model = keras.models.load_model(model_path, compile=False)
                    print(f"✅ Loaded {model_name}: {model_path}")
                    return model
                except Exception as e2:
                    print(f"❌ Error loading {model_name}: {str(e)}")
                    return None
        else:
            print(f"❌ Model not found: {model_path}")
            return None
    except Exception as e:
        print(f"❌ Error loading {model_name}: {str(e)}")
        return None

def compute_reconstruction_error_lstm_vae_gan(X_test, encoder_path, decoder_path):
    """Compute reconstruction error for LSTM-VAE-GAN"""
    encoder = load_model_safe(encoder_path, "LSTM-VAE-GAN encoder")
    decoder = load_model_safe(decoder_path, "LSTM-VAE-GAN decoder")
    
    if encoder is None or decoder is None:
        return None
    
    # Get latent layers
    z_mean_layer = None
    z_log_var_layer = None
    
    for layer in encoder.layers:
        if 'z_mean' in layer.name:
            z_mean_layer = layer
        elif 'z_log_var' in layer.name:
            z_log_var_layer = layer
    
    # Create intermediate model
    intermediate_model = tf.keras.Model(
        inputs=encoder.input,
        outputs=[z_mean_layer.output, z_log_var_layer.output]
    )
    z_mean, z_log_var = intermediate_model.predict(X_test, verbose=0)
    
    # Manual sampling
    epsilon = tf.random.normal(shape=tf.shape(z_mean), dtype=z_mean.dtype)
    z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
    z = z.numpy()
    
    # Reconstruct
    X_reconstructed = decoder.predict(z, verbose=0)
    
    # Compute MSE per sample
    mse_per_sample = np.mean(np.square(X_test - X_reconstructed), axis=(1, 2))
    return mse_per_sample

def compute_reconstruction_error_lstm_ae(X_test, model_path):
    """Compute reconstruction error for LSTM Autoencoder"""
    model = load_model_safe(model_path, "LSTM Autoencoder")
    if model is None:
        return None
    
    X_reconstructed = model.predict(X_test, verbose=0)
    mse_per_sample = np.mean(np.square(X_test - X_reconstructed), axis=(1, 2))
    return mse_per_sample

def compute_reconstruction_error_vae_gan(X_test, encoder_path, decoder_path):
    """Compute reconstruction error for VAE-GAN"""
    encoder = load_model_safe(encoder_path, "VAE-GAN encoder")
    decoder = load_model_safe(decoder_path, "VAE-GAN decoder")
    
    if encoder is None or decoder is None:
        return None
    
    latent_output = encoder.predict(X_test, verbose=0)
    
    if isinstance(latent_output, list) and len(latent_output) >= 3:
        z = latent_output[2]
    elif isinstance(latent_output, list) and len(latent_output) == 2:
        z_mean, z_log_var = latent_output
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
    else:
        z = latent_output
    
    X_reconstructed = decoder.predict(z, verbose=0)
    mse_per_sample = np.mean(np.square(X_test - X_reconstructed), axis=(1, 2))
    return mse_per_sample

def add_gaussian_noise(X, noise_level):
    """Add Gaussian noise to the data"""
    noise = np.random.normal(0, noise_level, X.shape)
    return X + noise

def scale_attack_magnitude(X, y, injection_factor):
    """Inject additional attack patterns with specified magnitude"""
    X_scaled = X.copy()
    attack_indices = np.where(y == 1)[0]
    
    # For each attack sample, inject additional anomalous patterns
    for idx in attack_indices:
        # Add random perturbations to simulate stronger attack injection
        perturbation = np.random.normal(0, injection_factor, X_scaled[idx].shape)
        X_scaled[idx] = X_scaled[idx] + perturbation
    
    return X_scaled

def get_precomputed_noise_robustness():
    """Get pre-computed noise robustness results"""
    # Data from your provided results
    noise_levels = [0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    
    # Recall scores for each model at different noise levels (converted from percentages)
    results = {
        'LSTM Autoencoder': {
            'noise_levels': noise_levels,
            'recall_scores': [90.87/100, 91.39/100, 92.12/100, 91.87/100, 92.37/100, 
                             92.27/100, 91.78/100, 93.28/100, 92.85/100, 92.47/100, 
                             92.41/100, 92.41/100, 92.41/100, 92.41/100],
            'pr_auc_scores': [90.83/100, 87.32/100, 88.06/100, 90.15/100, 94.88/100, 
                             94.54/100, 94.03/100, 96.62/100, 97.91/100, 98.74/100, 
                             98.74/100, 98.74/100, 98.74/100, 98.75/100]
        },
        'LSTM-GAN': {
            'noise_levels': noise_levels,
            'recall_scores': [63.99/100, 66.43/100, 64.75/100, 66.20/100, 63.34/100, 
                             64.36/100, 65.38/100, 63.18/100, 64.00/100, 64.40/100, 
                             64.43/100, 64.40/100, 64.40/100, 64.40/100],
            'pr_auc_scores': [90.73/100, 87.37/100, 87.86/100, 90.04/100, 95.21/100, 
                             94.32/100, 94.03/100, 96.59/100, 97.94/100, 98.70/100, 
                             98.70/100, 98.70/100, 98.70/100, 98.70/100]
        },
        'LSTM-VAE-GAN': {
            'noise_levels': noise_levels,
            'recall_scores': [100.00/100, 100.00/100, 100.00/100, 100.00/100, 100.00/100, 
                             100.00/100, 100.00/100, 100.00/100, 100.00/100, 100.00/100, 
                             100.00/100, 100.00/100, 100.00/100, 100.00/100],
            'pr_auc_scores': [88.55/100, 85.98/100, 86.55/100, 90.04/100, 94.47/100, 
                             94.70/100, 93.55/100, 95.86/100, 97.89/100, 98.80/100, 
                             98.80/100, 98.80/100, 98.80/100, 98.80/100]
        },
        'VAE-GAN': {
            'noise_levels': noise_levels,
            'recall_scores': [96.24/100, 96.59/100, 96.89/100, 96.81/100, 96.88/100, 
                             97.24/100, 96.56/100, 97.21/100, 97.40/100, 97.25/100, 
                             97.28/100, 97.28/100, 97.28/100, 97.28/100],
            'pr_auc_scores': [90.34/100, 87.14/100, 87.12/100, 90.70/100, 95.09/100, 
                             94.02/100, 94.23/100, 96.45/100, 97.97/100, 98.89/100, 
                             98.90/100, 98.90/100, 98.90/100, 98.90/100]
        }
    }
    
    return results

def get_precomputed_attack_robustness():
    """Get pre-computed attack magnitude robustness results"""
    # Data from your provided results  
    attack_levels = [0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    
    # Recall scores for each model at different attack levels (converted from percentages)
    results = {
        'LSTM Autoencoder': {
            'magnitude_scales': attack_levels,
            'recall_scores': [92.22/100, 92.18/100, 92.12/100, 91.92/100, 91.96/100, 
                             91.96/100, 91.14/100, 89.47/100, 87.57/100, 85.28/100, 
                             81.39/100, 76.68/100, 71.32/100, 65.27/100],
            'pr_auc_scores': [87.83/100, 87.90/100, 88.06/100, 88.31/100, 88.65/100, 
                             88.84/100, 90.03/100, 91.31/100, 92.49/100, 93.48/100, 
                             94.28/100, 94.93/100, 95.48/100, 95.93/100]
        },
        'LSTM-GAN': {
            'magnitude_scales': attack_levels,
            'recall_scores': [64.62/100, 64.58/100, 64.75/100, 64.39/100, 64.26/100, 
                             64.45/100, 63.90/100, 62.56/100, 60.30/100, 57.55/100, 
                             53.70/100, 49.51/100, 45.45/100, 42.05/100],
            'pr_auc_scores': [87.64/100, 87.70/100, 87.86/100, 88.11/100, 88.44/100, 
                             88.66/100, 89.85/100, 91.14/100, 92.34/100, 93.35/100, 
                             94.17/100, 94.84/100, 95.39/100, 95.85/100]
        },
        'LSTM-VAE-GAN': {
            'magnitude_scales': attack_levels,
            'recall_scores': [100.00/100, 100.00/100, 100.00/100, 100.00/100, 100.00/100, 
                             100.00/100, 100.00/100, 100.00/100, 100.00/100, 100.00/100, 
                             100.00/100, 100.00/100, 100.00/100, 100.00/100],
            'pr_auc_scores': [86.43/100, 86.47/100, 86.55/100, 86.67/100, 86.83/100, 
                             86.93/100, 87.59/100, 88.47/100, 89.48/100, 90.48/100, 
                             91.40/100, 92.24/100, 92.96/100, 93.58/100]
        },
        'VAE-GAN': {
            'magnitude_scales': attack_levels,
            'recall_scores': [96.83/100, 96.89/100, 96.89/100, 96.83/100, 96.86/100, 
                             96.76/100, 96.50/100, 95.95/100, 95.39/100, 94.05/100, 
                             92.22/100, 89.80/100, 86.04/100, 82.15/100],
            'pr_auc_scores': [86.93/100, 86.98/100, 87.12/100, 87.32/100, 87.61/100, 
                             87.79/100, 88.95/100, 90.24/100, 91.49/100, 92.58/100, 
                             93.48/100, 94.21/100, 94.81/100, 95.30/100]
        }
    }
    
    return results

def evaluate_noise_robustness():
    """Use pre-computed noise robustness results"""
    print("\n" + "="*80)
    print("NOISE ROBUSTNESS EVALUATION")
    print("="*80)
    print("Using pre-computed results...")
    
    return get_precomputed_noise_robustness()

def evaluate_attack_magnitude_robustness():
    """Use pre-computed attack magnitude robustness results"""
    print("\n" + "="*80)
    print("ATTACK MAGNITUDE ROBUSTNESS EVALUATION")
    print("="*80)
    print("Using pre-computed results...")
    
    return get_precomputed_attack_robustness()

def plot_robustness_results(noise_results, attack_results, save_path='outputs/robustness_evaluation.png'):
    """Plot robustness results for both noise and attack injection"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Define colors for each model
    colors = {'LSTM Autoencoder': '#1f77b4', 'LSTM-GAN': '#ff7f0e', 
              'LSTM-VAE-GAN': '#2ca02c', 'VAE-GAN': '#d62728'}
    
    # Plot noise robustness (Recall)
    for model_name, data in noise_results.items():
        ax1.plot(data['noise_levels'], [r*100 for r in data['recall_scores']], 
                marker='o', linewidth=2, label=model_name, color=colors[model_name])
    
    ax1.set_xlabel('Noise Level')
    ax1.set_ylabel('Recall (%)')
    ax1.set_title('Model Robustness vs Noise Level (Recall, Threshold = 0.95)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 105)
    
    # Plot attack injection robustness (Recall)
    for model_name, data in attack_results.items():
        ax2.plot(data['magnitude_scales'], [r*100 for r in data['recall_scores']], 
                marker='s', linewidth=2, label=model_name, color=colors[model_name])
    
    ax2.set_xlabel('Attack Magnitude Scale')
    ax2.set_ylabel('Recall (%)')
    ax2.set_title('Model Robustness vs Attack Injection Level (Recall, Threshold = 0.95)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_pr_auc_robustness_results(noise_results, attack_results, save_path='outputs/pr_auc_robustness_evaluation.png'):
    """Plot PR-AUC robustness results for both noise and attack injection"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Define colors for each model
    colors = {'LSTM Autoencoder': '#1f77b4', 'LSTM-GAN': '#ff7f0e', 
              'LSTM-VAE-GAN': '#2ca02c', 'VAE-GAN': '#d62728'}
    
    # Plot noise robustness (PR-AUC)
    for model_name, data in noise_results.items():
        ax1.plot(data['noise_levels'], [r*100 for r in data['pr_auc_scores']], 
                marker='o', linewidth=2, label=model_name, color=colors[model_name])
    
    ax1.set_xlabel('Noise Level')
    ax1.set_ylabel('PR-AUC (%)')
    ax1.set_title('Model Robustness vs Noise Level (PR-AUC)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(80, 100)
    
    # Plot attack injection robustness (PR-AUC)
    for model_name, data in attack_results.items():
        ax2.plot(data['magnitude_scales'], [r*100 for r in data['pr_auc_scores']], 
                marker='s', linewidth=2, label=model_name, color=colors[model_name])
    
    ax2.set_xlabel('Attack Magnitude Scale')
    ax2.set_ylabel('PR-AUC (%)')
    ax2.set_title('Model Robustness vs Attack Injection Level (PR-AUC)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(80, 100)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def save_results(noise_results, magnitude_results):
    """Save results to JSON files"""
    
    # Save noise robustness results
    noise_path = 'outputs/noise_robustness_results.json'
    with open(noise_path, 'w') as f:
        json.dump(noise_results, f, indent=2)
    print(f"💾 Noise robustness results saved to: {noise_path}")
    
    # Save attack magnitude robustness results
    magnitude_path = 'outputs/attack_magnitude_robustness_results.json'
    with open(magnitude_path, 'w') as f:
        json.dump(magnitude_results, f, indent=2)
    print(f"💾 Attack magnitude robustness results saved to: {magnitude_path}")

def main():
    """Main function to run robustness evaluation"""
    print("🔄 Loading pre-computed robustness results...")
    
    # Get pre-computed results
    noise_results = get_precomputed_noise_robustness()
    attack_results = get_precomputed_attack_robustness()
    
    # Create plots
    print("📊 Creating recall robustness plots...")
    plot_robustness_results(noise_results, attack_results)
    
    print("📊 Creating PR-AUC robustness plots...")
    plot_pr_auc_robustness_results(noise_results, attack_results)
    
    # Save results to JSON
    print("💾 Saving results...")
    
    # Convert numpy arrays to lists for JSON serialization
    noise_json = {}
    attack_json = {}
    
    for model_name, data in noise_results.items():
        noise_json[model_name] = {
            'noise_levels': data['noise_levels'],
            'recall_scores': data['recall_scores'],
            'pr_auc_scores': data['pr_auc_scores']
        }
    
    for model_name, data in attack_results.items():
        attack_json[model_name] = {
            'magnitude_scales': data['magnitude_scales'],
            'recall_scores': data['recall_scores'],
            'pr_auc_scores': data['pr_auc_scores']
        }
    
    # Save noise robustness results
    with open('outputs/noise_robustness_results.json', 'w') as f:
        json.dump(noise_json, f, indent=2)
    
    # Save attack magnitude robustness results
    with open('outputs/attack_magnitude_robustness_results.json', 'w') as f:
        json.dump(attack_json, f, indent=2)
    
    print("✅ Robustness evaluation completed!")
    print(f"📁 Results saved to outputs/")
    print(f"📊 Recall plots saved to outputs/robustness_evaluation.png")
    print(f"📊 PR-AUC plots saved to outputs/pr_auc_robustness_evaluation.png")

if __name__ == "__main__":
    main()
