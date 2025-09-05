import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import os
import sys
from sklearn.metrics import precision_recall_curve, auc, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# Add the project root to Python path
sys.path.append('/home/brianyudiva/Documents/Project/lstm-vae-gan-smart-grid')

from utils.utils import create_anomaly_labels
from models.baseline_models import SamplingLayer  # Import the custom layer


def load_data():
    """Load the testing data"""
    data_path = '/home/brianyudiva/Documents/Project/lstm-vae-gan-smart-grid/data/sequences'
    
    X_test = np.load(os.path.join(data_path, 'X_test.npy'))
    y_test = np.load(os.path.join(data_path, 'y_test_binary.npy'))
    
    print(f"Test data loaded:")
    print(f"  X_test shape: {X_test.shape}")
    print(f"  y_test shape: {y_test.shape}")
    print(f"  Normal samples: {np.sum(y_test == 0)}")
    print(f"  Anomaly samples: {np.sum(y_test == 1)}")
    
    return X_test, y_test


def load_model_safe(model_path, model_name=""):
    """Safely load a model with error handling and custom objects"""
    try:
        if os.path.exists(model_path):
            # Define custom objects for models with VAE components
            custom_objects = {
                'SamplingLayer': SamplingLayer,
                'sampling': SamplingLayer,  # For legacy compatibility
            }
            
            # Try loading with custom objects first
            try:
                model = keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
                print(f"✅ Loaded {model_name}: {model_path}")
                return model
            except Exception as e:
                # If that fails, try without custom objects (for simpler models)
                try:
                    model = keras.models.load_model(model_path, compile=False)
                    print(f"✅ Loaded {model_name}: {model_path}")
                    return model
                except Exception as e2:
                    print(f"❌ Error loading {model_name}: {str(e)}")
                    print(f"   Secondary error: {str(e2)}")
                    return None
        else:
            print(f"❌ Model not found: {model_path}")
            return None
    except Exception as e:
        print(f"❌ Error loading {model_name}: {str(e)}")
        return None


def detect_model_type_and_latent_dim(model_name):
    """Detect model configuration from available files"""
    
    # Try to load best hyperparameters for latent dimension
    best_params_path = '/home/brianyudiva/Documents/Project/lstm-vae-gan-smart-grid/best_hyperparameters.json'
    latent_dim = 32  # default
    
    if os.path.exists(best_params_path):
        try:
            with open(best_params_path, 'r') as f:
                params = json.load(f)
                latent_dim = params.get('latent_dim', 32)
        except:
            pass
    
    return latent_dim


def compute_reconstruction_error(X_test, model, model_type, encoder_path=None, decoder_path=None):
    """Compute reconstruction error for anomaly detection"""
    
    try:
        if model_type in ['lstm_autoencoder', 'lstm_gan_generator']:
            # Direct reconstruction models
            X_reconstructed = model.predict(X_test, verbose=0)
        elif model_type in ['vae_gan_full', 'lstm_vae_full']:
            # VAE models - try direct reconstruction first
            try:
                X_reconstructed = model.predict(X_test, verbose=0)
            except Exception as e:
                print(f"   Direct VAE prediction failed, trying encoder-decoder approach: {str(e)}")
                # If VAE full model fails, try encoder-decoder approach
                if encoder_path and decoder_path:
                    encoder = load_model_safe(encoder_path, f"{model_type}_encoder")
                    decoder = load_model_safe(decoder_path, f"{model_type}_decoder")
                    if encoder and decoder:
                        # Get latent representation
                        latent_output = encoder.predict(X_test, verbose=0)
                        if isinstance(latent_output, list):
                            # For VAE: [z_mean, z_log_var, z] - use sampled z
                            z = latent_output[2]
                        else:
                            z = latent_output
                        # Reconstruct
                        X_reconstructed = decoder.predict(z, verbose=0)
                    else:
                        return None
                else:
                    return None
        elif model_type == 'lstm_vae_gan':
            # Original LSTM-VAE-GAN - try encoder-decoder approach
            if encoder_path and decoder_path:
                encoder = load_model_safe(encoder_path, f"{model_type}_encoder")
                decoder = load_model_safe(decoder_path, f"{model_type}_decoder")
                if encoder and decoder:
                    latent_output = encoder.predict(X_test, verbose=0)
                    if isinstance(latent_output, list):
                        z = latent_output[2]  # Use sampled z
                    else:
                        z = latent_output
                    X_reconstructed = decoder.predict(z, verbose=0)
                else:
                    return None
            else:
                # Try direct prediction as fallback
                X_reconstructed = model.predict(X_test, verbose=0)
        else:
            print(f"Unknown model type for reconstruction: {model_type}")
            return None
            
        # Compute MSE per sample
        mse_per_sample = np.mean(np.square(X_test - X_reconstructed), axis=(1, 2))
        return mse_per_sample
        
    except Exception as e:
        print(f"Error computing reconstruction error for {model_type}: {str(e)}")
        return None


def evaluate_model(model_path, model_name, model_type, X_test, y_test, encoder_path=None, decoder_path=None):
    """Evaluate a single model"""
    
    print(f"\n{'='*60}")
    print(f"EVALUATING: {model_name}")
    print(f"{'='*60}")
    
    # Load model
    model = load_model_safe(model_path, model_name)
    if model is None:
        return None
    
    # Get reconstruction errors
    reconstruction_errors = compute_reconstruction_error(X_test, model, model_type, encoder_path, decoder_path)
    if reconstruction_errors is None:
        return None
    
    # Compute metrics using reconstruction error as anomaly score
    # Higher reconstruction error = more likely to be anomaly
    anomaly_scores = reconstruction_errors
    
    # Compute Precision-Recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, anomaly_scores)
    pr_auc = auc(recall, precision)
    
    # Find optimal threshold (best F1 score)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else np.median(anomaly_scores)
    
    # Compute metrics at optimal threshold
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
        'model_params': model.count_params(),
        'mean_reconstruction_error': np.mean(reconstruction_errors),
        'std_reconstruction_error': np.std(reconstruction_errors)
    }
    
    # Print results in percentage format
    print(f"PR-AUC:     {pr_auc*100:.2f}%")
    print(f"Accuracy:   {accuracy*100:.2f}%")
    print(f"Precision:  {precision_opt*100:.2f}%")
    print(f"Recall:     {recall_opt*100:.2f}%")
    print(f"F1-Score:   {f1_opt*100:.2f}%")
    print(f"Parameters: {model.count_params():,}")
    print(f"Optimal Threshold: {optimal_threshold:.6f}")
    
    return results


def create_comparison_table(all_results):
    """Create a comparison table of all models"""
    
    if not all_results:
        print("No results to compare")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Sort by PR-AUC (best first)
    df = df.sort_values('pr_auc', ascending=False)
    
    print(f"\n{'='*120}")
    print("MODEL COMPARISON RESULTS")
    print(f"{'='*120}")
    
    # Print formatted table
    print(f"{'Model':<20} {'PR-AUC':<8} {'Accuracy':<9} {'Precision':<10} {'Recall':<8} {'F1-Score':<9} {'Parameters':<12}")
    print("-" * 120)
    
    for _, row in df.iterrows():
        print(f"{row['model_name']:<20} "
              f"{row['pr_auc']*100:>6.2f}%  "
              f"{row['accuracy']*100:>7.2f}%  "
              f"{row['precision']*100:>8.2f}%  "
              f"{row['recall']*100:>6.2f}%  "
              f"{row['f1_score']*100:>7.2f}%  "
              f"{row['model_params']:>10,}")
    
    print("-" * 120)
    
    # Performance ranking
    print(f"\n🏆 RANKING BY PR-AUC:")
    for i, (_, row) in enumerate(df.iterrows(), 1):
        print(f"{i}. {row['model_name']}: {row['pr_auc']*100:.2f}%")
    
    return df


def plot_comparison_charts(all_results):
    """Create visualization charts comparing all models"""
    
    if not all_results:
        return
    
    df = pd.DataFrame(all_results)
    model_names = df['model_name'].tolist()
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Baseline Models Comparison', fontsize=16, fontweight='bold')
    
    # PR-AUC comparison
    axes[0, 0].bar(model_names, df['pr_auc'] * 100)
    axes[0, 0].set_title('PR-AUC Comparison (%)')
    axes[0, 0].set_ylabel('PR-AUC (%)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # F1-Score comparison  
    axes[0, 1].bar(model_names, df['f1_score'] * 100)
    axes[0, 1].set_title('F1-Score Comparison (%)')
    axes[0, 1].set_ylabel('F1-Score (%)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Parameter count comparison
    axes[1, 0].bar(model_names, df['model_params'])
    axes[1, 0].set_title('Model Parameters')
    axes[1, 0].set_ylabel('Number of Parameters')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Multi-metric comparison
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    x = np.arange(len(model_names))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        axes[1, 1].bar(x + i*width, df[metric] * 100, width, label=metric.capitalize())
    
    axes[1, 1].set_title('All Metrics Comparison (%)')
    axes[1, 1].set_ylabel('Score (%)')
    axes[1, 1].set_xticks(x + width * 1.5)
    axes[1, 1].set_xticklabels(model_names, rotation=45)
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    # Save plot
    plot_path = '/home/brianyudiva/Documents/Project/lstm-vae-gan-smart-grid/outputs/baseline_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Comparison plots saved to: {plot_path}")
    
    plt.show()


def main():
    """Main evaluation function"""
    
    print("BASELINE MODELS EVALUATION")
    print("="*80)
    
    # Load test data
    X_test, y_test = load_data()
    
    # Define models to evaluate with encoder/decoder paths for VAE models
    checkpoints_dir = '/home/brianyudiva/Documents/Project/lstm-vae-gan-smart-grid/outputs/checkpoints'
    
    models_to_evaluate = [
        # Original LSTM-VAE-GAN (for comparison)
        {
            'model_path': f'{checkpoints_dir}/lstm_vae_gan_encoder.h5',
            'model_name': 'LSTM-VAE-GAN (Original)',
            'model_type': 'lstm_vae_gan',
            'encoder_path': f'{checkpoints_dir}/lstm_vae_gan_encoder.h5',
            'decoder_path': f'{checkpoints_dir}/lstm_vae_gan_decoder.h5'
        },
        
        # Baseline models
        {
            'model_path': f'{checkpoints_dir}/lstm_autoencoder_full.h5',
            'model_name': 'LSTM Autoencoder',
            'model_type': 'lstm_autoencoder',
            'encoder_path': None,
            'decoder_path': None
        },
        {
            'model_path': f'{checkpoints_dir}/vae_gan_full.h5',
            'model_name': 'VAE-GAN',
            'model_type': 'vae_gan_full',
            'encoder_path': f'{checkpoints_dir}/vae_gan_encoder.h5',
            'decoder_path': f'{checkpoints_dir}/vae_gan_decoder.h5'
        },
        {
            'model_path': f'{checkpoints_dir}/lstm_gan_generator.h5',
            'model_name': 'LSTM-GAN',
            'model_type': 'lstm_gan_generator',
            'encoder_path': None,
            'decoder_path': None
        },
        {
            'model_path': f'{checkpoints_dir}/lstm_vae_full.h5',
            'model_name': 'LSTM-VAE',
            'model_type': 'lstm_vae_full',
            'encoder_path': f'{checkpoints_dir}/lstm_vae_encoder.h5',
            'decoder_path': f'{checkpoints_dir}/lstm_vae_decoder.h5'
        },
    ]
    
    # Special handling for original LSTM-VAE-GAN if available
    if not os.path.exists(f'{checkpoints_dir}/lstm_vae_gan_encoder.h5'):
        # Try alternative naming from Optuna optimization
        if os.path.exists(f'{checkpoints_dir}/lstm_vae_gan_quick_optuna_encoder.h5'):
            models_to_evaluate[0]['model_path'] = f'{checkpoints_dir}/lstm_vae_gan_quick_optuna_encoder.h5'
            models_to_evaluate[0]['model_name'] = 'LSTM-VAE-GAN (Optuna)'
            models_to_evaluate[0]['encoder_path'] = f'{checkpoints_dir}/lstm_vae_gan_quick_optuna_encoder.h5'
            models_to_evaluate[0]['decoder_path'] = f'{checkpoints_dir}/lstm_vae_gan_quick_optuna_decoder.h5'
        else:
            print("⚠️  Original LSTM-VAE-GAN model not found, skipping...")
            models_to_evaluate = models_to_evaluate[1:]  # Skip first model
    
    # Evaluate all models
    all_results = []
    
    for model_config in models_to_evaluate:
        result = evaluate_model(
            model_config['model_path'],
            model_config['model_name'],
            model_config['model_type'],
            X_test, y_test,
            model_config.get('encoder_path'),
            model_config.get('decoder_path')
        )
        if result is not None:
            all_results.append(result)
    
    # Create comparison
    if all_results:
        comparison_df = create_comparison_table(all_results)
        
        # Save results
        results_path = f'{checkpoints_dir}/baseline_evaluation_results.json'
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        if comparison_df is not None:
            csv_path = f'{checkpoints_dir}/baseline_comparison.csv'
            comparison_df.to_csv(csv_path, index=False)
            print(f"\n💾 Results saved to:")
            print(f"   JSON: {results_path}")
            print(f"   CSV:  {csv_path}")
        
        # Create plots
        try:
            plot_comparison_charts(all_results)
        except Exception as e:
            print(f"⚠️  Could not create plots: {str(e)}")
    
    else:
        print("❌ No models could be evaluated successfully")
    
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
