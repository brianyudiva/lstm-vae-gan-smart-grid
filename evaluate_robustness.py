import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import json
import os
import sys
from models.baseline_models import SamplingLayer

sys.path.append('.')

def set_reproducible_seeds(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    try:
        tf.compat.v1.set_random_seed(seed)
    except:
        pass

def load_model(model_path, model_name=""):
    custom_objects = {
        'SamplingLayer': SamplingLayer,
        'sampling': SamplingLayer,
    }
    
    try:
        model = keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
        return model
    except Exception as e:
        model = keras.models.load_model(model_path, compile=False)
        return model

def get_precomputed_noise_robustness():
    noise_levels = [0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    
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
    attack_levels = [0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    
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

def plot_robustness_results(noise_results, attack_results):
    colors = {'LSTM Autoencoder': '#1f77b4', 'LSTM-GAN': '#ff7f0e', 
              'LSTM-VAE-GAN': '#2ca02c', 'VAE-GAN': '#d62728'}
    
    # Plot 1: Noise robustness (Recall)
    plt.figure(figsize=(10, 6))
    for model_name, data in noise_results.items():
        plt.plot(data['noise_levels'], [r*100 for r in data['recall_scores']], 
                marker='o', linewidth=2, label=model_name, color=colors[model_name])
    
    plt.xlabel('Noise Level')
    plt.ylabel('Recall (%)')
    plt.title('Model Robustness vs Noise Level (Recall, Threshold = 0.95)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 105)
    plt.tight_layout()
    plt.savefig('outputs/recall_noise_robustness.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 2: Attack injection robustness (Recall)
    plt.figure(figsize=(10, 6))
    for model_name, data in attack_results.items():
        plt.plot(data['magnitude_scales'], [r*100 for r in data['recall_scores']], 
                marker='s', linewidth=2, label=model_name, color=colors[model_name])
    
    plt.xlabel('Attack Magnitude Scale')
    plt.ylabel('Recall (%)')
    plt.title('Model Robustness vs Attack Injection Level (Recall, Threshold = 0.95)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 105)
    plt.tight_layout()
    plt.savefig('outputs/recall_attack_robustness.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_pr_auc_robustness_results(noise_results, attack_results):
    colors = {'LSTM Autoencoder': '#1f77b4', 'LSTM-GAN': '#ff7f0e', 
              'LSTM-VAE-GAN': '#2ca02c', 'VAE-GAN': '#d62728'}
    
    # Plot 1: Noise robustness (PR-AUC)
    plt.figure(figsize=(10, 6))
    for model_name, data in noise_results.items():
        plt.plot(data['noise_levels'], [r*100 for r in data['pr_auc_scores']], 
                marker='o', linewidth=2, label=model_name, color=colors[model_name])
    
    plt.xlabel('Noise Level')
    plt.ylabel('PR-AUC (%)')
    plt.title('Model Robustness vs Noise Level (PR-AUC)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(80, 100)
    plt.tight_layout()
    plt.savefig('outputs/pr_auc_noise_robustness.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 2: Attack injection robustness (PR-AUC)
    plt.figure(figsize=(10, 6))
    for model_name, data in attack_results.items():
        plt.plot(data['magnitude_scales'], [r*100 for r in data['pr_auc_scores']], 
                marker='s', linewidth=2, label=model_name, color=colors[model_name])
    
    plt.xlabel('Attack Magnitude Scale')
    plt.ylabel('PR-AUC (%)')
    plt.title('Model Robustness vs Attack Injection Level (PR-AUC)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(80, 100)
    plt.tight_layout()
    plt.savefig('outputs/pr_auc_attack_robustness.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():    
    noise_results = get_precomputed_noise_robustness()
    attack_results = get_precomputed_attack_robustness()
    
    plot_robustness_results(noise_results, attack_results)
    plot_pr_auc_robustness_results(noise_results, attack_results)
    
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
    
    with open('outputs/noise_robustness_results.json', 'w') as f:
        json.dump(noise_json, f, indent=2)
    
    with open('outputs/attack_magnitude_robustness_results.json', 'w') as f:
        json.dump(attack_json, f, indent=2)

if __name__ == "__main__":
    main()
