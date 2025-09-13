import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import time
from sklearn.metrics import precision_recall_curve, auc, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import pandas as pd
from models.baseline_models import SamplingLayer
import random

seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

def load_data():
    sequence_dir = 'data/sequences'
    
    X_test = np.load(f"{sequence_dir}/X_test.npy")
    y_test = np.load(f"{sequence_dir}/y_test_binary.npy")
    
    print(f"Test data loaded:")
    print(f"  X_test shape: {X_test.shape}")
    print(f"  y_test shape: {y_test.shape}")
    
    return X_test, y_test


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

def compute_reconstruction_error(X_test, model, model_type, encoder_path=None, decoder_path=None):
    start_time = time.time()
    
    if model_type in ['lstm_autoencoder', 'lstm_gan_generator']:
        X_reconstructed = model.predict(X_test, verbose=0)

    elif model_type in ['vae_gan_full']:
        encoder = load_model(encoder_path, f"{model_type}_encoder")
        decoder = load_model(decoder_path, f"{model_type}_decoder")

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

    elif model_type == 'lstm_vae_gan':
        encoder = load_model(encoder_path, f"{model_type}_encoder")
        decoder = load_model(decoder_path, f"{model_type}_decoder")

        z_mean_layer = None
        z_log_var_layer = None
        
        for layer in encoder.layers:
            if 'z_mean' in layer.name:
                z_mean_layer = layer
            elif 'z_log_var' in layer.name:
                z_log_var_layer = layer
        
        intermediate_model = tf.keras.Model(inputs=encoder.input, outputs=[z_mean_layer.output, z_log_var_layer.output])
        z_mean, z_log_var = intermediate_model.predict(X_test, verbose=0)
        
        epsilon = tf.random.normal(shape=tf.shape(z_mean), dtype=z_mean.dtype)
        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
        z = z.numpy()
        
        X_reconstructed = decoder.predict(z, verbose=0)
        
    detection_time = time.time() - start_time
    
    mse_per_sample = np.mean(np.square(X_test - X_reconstructed), axis=(1, 2))
    return mse_per_sample, detection_time

def evaluate_model(model_path, model_name, model_type, X_test, y_test, encoder_path=None, decoder_path=None, discriminator_path=None):    
    print(f"\n{'='*60}")
    print(f"EVALUATING: {model_name}")
    print(f"{'='*60}")
    
    model = load_model(model_path, model_name)
    
    reconstruction_errors, detection_time = compute_reconstruction_error(X_test, model, model_type, encoder_path, decoder_path)
    
    if reconstruction_errors is None:
        return None
    
    anomaly_scores = reconstruction_errors
    
    samples_per_second = len(X_test) / detection_time
    milliseconds_per_sample = (detection_time * 1000) / len(X_test)
    
    precision, recall, thresholds = precision_recall_curve(y_test, anomaly_scores)
    pr_auc = auc(recall, precision)
    
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    # optimal_threshold = 0.95 # Uncomment to use a fixed threshold
    
    y_pred = (anomaly_scores > optimal_threshold).astype(int)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision_opt = precision_score(y_test, y_pred, zero_division=0)
    recall_opt = recall_score(y_test, y_pred, zero_division=0)
    f1_opt = f1_score(y_test, y_pred, zero_division=0)
    
    results = {
        'model_name': model_name,
        'model_type': model_type,
        'pr_auc': pr_auc,
        'optimal_threshold': optimal_threshold,
        'accuracy': accuracy,
        'precision': precision_opt,
        'recall': recall_opt,
        'f1_score': f1_opt,
        'mean_reconstruction_error': np.mean(reconstruction_errors),
        'std_reconstruction_error': np.std(reconstruction_errors),
        'detection_time_seconds': detection_time,
        'samples_per_second': samples_per_second,
        'milliseconds_per_sample': milliseconds_per_sample
    }
    
    print(f"PR-AUC:     {pr_auc*100:.4f}%")
    print(f"Accuracy:   {accuracy*100:.4f}%")
    print(f"Precision:  {precision_opt*100:.4f}%")
    print(f"Recall:     {recall_opt*100:.4f}%")
    print(f"F1-Score:   {f1_opt*100:.4f}%")
    print(f"Optimal Threshold: {optimal_threshold:.6f}")
    
    return results


def create_comparison_table(all_results):    
    df = pd.DataFrame(all_results)
    
    df = df.sort_values('model_name', ascending=True)
    
    print(f"\n{'='*50}")
    print("MODEL COMPARISON RESULTS")
    print(f"{'='*50}")
    
    print(f"{'Model':<20} {'PR-AUC':<8} {'Accuracy':<9} {'Precision':<10} {'Recall':<8} {'F1-Score':<9}")

    print("-" * 50)
    
    for _, row in df.iterrows():
        print(f"{row['model_name']:<20} "
              f"{row['pr_auc']*100:>6.2f}%  "
              f"{row['accuracy']*100:>7.2f}%  "
              f"{row['precision']*100:>8.2f}%  "
              f"{row['recall']*100:>6.2f}%  "
              f"{row['f1_score']*100:>7.2f}%  "
              )
    
    print("-" * 50)
    
    print(f"\nRANKING BY PR-AUC:")
    for i, (_, row) in enumerate(df.iterrows(), 1):
        print(f"{i}. {row['model_name']}: {row['pr_auc']*100:.3f}%")
    
    return df

def plot_pr_curves(all_results, X_test, y_test):
    plt.style.use('seaborn-v0_8')
    plt.figure(figsize=(12, 8))
    
    checkpoints_dir = 'outputs/checkpoints'
    
    models_config = [
        {
            'model_path': f'{checkpoints_dir}/lstm_vae_gan_encoder.h5',
            'model_name': 'LSTM-VAE-GAN',
            'model_type': 'lstm_vae_gan',
            'encoder_path': f'{checkpoints_dir}/lstm_vae_gan_encoder.h5',
            'decoder_path': f'{checkpoints_dir}/lstm_vae_gan_decoder.h5',
            'color': '#d62728'
        },
        {
            'model_path': f'{checkpoints_dir}/lstm_autoencoder_full.h5',
            'model_name': 'LSTM Autoencoder',
            'model_type': 'lstm_autoencoder',
            'encoder_path': None,
            'decoder_path': None,
            'color': '#2ca02c'
        },
        {
            'model_path': f'{checkpoints_dir}/vae_gan_full.h5',
            'model_name': 'VAE-GAN',
            'model_type': 'vae_gan_full',
            'encoder_path': f'{checkpoints_dir}/vae_gan_encoder.h5',
            'decoder_path': f'{checkpoints_dir}/vae_gan_decoder.h5',
            'color': '#ff7f0e'
        },
        {
            'model_path': f'{checkpoints_dir}/lstm_gan_generator.h5',
            'model_name': 'LSTM-GAN',
            'model_type': 'lstm_gan_generator',
            'encoder_path': None,
            'decoder_path': None,
            'color': '#1f77b4'
        }
    ]
    
    for model_config in models_config:        
        model = load_model(model_config['model_path'], model_config['model_name'])

        reconstruction_errors, _ = compute_reconstruction_error(
            X_test, model, model_config['model_type'], 
            model_config.get('encoder_path'), model_config.get('decoder_path')
        )
        
        precision, recall, thresholds = precision_recall_curve(y_test, reconstruction_errors)
        pr_auc = auc(recall, precision)
        
        plt.plot(recall, precision, 
                color=model_config['color'], 
                linewidth=2.5, 
                label=f"{model_config['model_name']} (AUC = {pr_auc:.3f})")
    
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('Precision-Recall Curves - FDIA Detection Models', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    plt.tight_layout()
    
    output_path = 'outputs/precision_recall_curves.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.show()

    return None

def main():    
    print("BASELINE MODELS EVALUATION")
    print("="*80)
    
    X_test, y_test = load_data()
    
    checkpoints_dir = 'outputs/checkpoints'
    
    models_to_evaluate = [
        {
            'model_path': f'{checkpoints_dir}/lstm_vae_gan_encoder.h5',
            'model_name': 'LSTM-VAE-GAN',
            'model_type': 'lstm_vae_gan',
            'encoder_path': f'{checkpoints_dir}/lstm_vae_gan_encoder.h5',
            'decoder_path': f'{checkpoints_dir}/lstm_vae_gan_decoder.h5',
            'discriminator_path': f'{checkpoints_dir}/lstm_vae_gan_discriminator.h5'
        },
        {
            'model_path': f'{checkpoints_dir}/lstm_autoencoder_full.h5',
            'model_name': 'LSTM Autoencoder',
            'model_type': 'lstm_autoencoder',
            'encoder_path': None,
            'decoder_path': None,
            'discriminator_path': None
        },
        {
            'model_path': f'{checkpoints_dir}/vae_gan_full.h5',
            'model_name': 'VAE-GAN',
            'model_type': 'vae_gan_full',
            'encoder_path': f'{checkpoints_dir}/vae_gan_encoder.h5',
            'decoder_path': f'{checkpoints_dir}/vae_gan_decoder.h5',
            'discriminator_path': f'{checkpoints_dir}/vae_gan_discriminator.h5'
        },
        {
            'model_path': f'{checkpoints_dir}/lstm_gan_generator.h5',
            'model_name': 'LSTM-GAN',
            'model_type': 'lstm_gan_generator',
            'encoder_path': None,
            'decoder_path': None,
            'discriminator_path': f'{checkpoints_dir}/lstm_gan_discriminator.h5'
        },
    ]

    all_results = []
    
    for model_config in models_to_evaluate:
        result = evaluate_model(
            model_config['model_path'],
            model_config['model_name'],
            model_config['model_type'],
            X_test, y_test,
            model_config.get('encoder_path'),
            model_config.get('decoder_path'),
            model_config.get('discriminator_path')
        )
        if result is not None:
            all_results.append(result)
    
    comparison_df = create_comparison_table(all_results)
    
    results_path = f'{checkpoints_dir}/baseline_evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    csv_path = f'{checkpoints_dir}/baseline_comparison.csv'
    comparison_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to:")
    print(f"   JSON: {results_path}")
    print(f"   CSV:  {csv_path}")

    plot_pr_curves(all_results, X_test, y_test)
    
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
