import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import os
import time
from sklearn.metrics import precision_recall_curve, auc, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import pandas as pd
from models.baseline_models import SamplingLayer  # Import the custom layer
import random


seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

def load_data():
    """Load the testing data"""
    data_path = '/home/brianyudiva/Documents/Project/lstm-vae-gan-smart-grid/data/sequences'
    
    X_test = np.load(os.path.join(data_path, 'X_test.npy'))
    y_test = np.load(os.path.join(data_path, 'y_test_binary.npy'))
    
    print(f"Test data loaded:")
    print(f"  X_test shape: {X_test.shape}")
    print(f"  y_test shape: {y_test.shape}")
    
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
    # Try to load best hyperparameters for latent dimension
    best_params_path = '/home/brianyudiva/Documents/Project/lstm-vae-gan-smart-grid/best_hyperparameters.json'
    latent_dim = 16  # default
    
    if os.path.exists(best_params_path):
        try:
            with open(best_params_path, 'r') as f:
                params = json.load(f)
                latent_dim = params.get('latent_dim', 16)
        except:
            pass
    
    return latent_dim


def compute_reconstruction_error(X_test, model, model_type, encoder_path=None, decoder_path=None):
    """Compute reconstruction error for anomaly detection"""
    
    try:
        # Start timing
        start_time = time.time()
        
        if model_type in ['lstm_autoencoder', 'lstm_gan_generator']:
            X_reconstructed = model.predict(X_test, verbose=0)
        elif model_type in ['vae_gan_full']:
            encoder = load_model_safe(encoder_path, f"{model_type}_encoder")
            decoder = load_model_safe(decoder_path, f"{model_type}_decoder")

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

            encoder = load_model_safe(encoder_path, f"{model_type}_encoder")
            decoder = load_model_safe(decoder_path, f"{model_type}_decoder")

            z_mean_layer = None
            z_log_var_layer = None
            
            for layer in encoder.layers:
                if 'z_mean' in layer.name:
                    z_mean_layer = layer
                elif 'z_log_var' in layer.name:
                    z_log_var_layer = layer
            
            # Create intermediate model up to z_mean and z_log_var
            intermediate_model = tf.keras.Model(
                inputs=encoder.input,
                outputs=[z_mean_layer.output, z_log_var_layer.output]
            )
            z_mean, z_log_var = intermediate_model.predict(X_test, verbose=0)
            
            # Manual sampling
            epsilon = tf.random.normal(shape=tf.shape(z_mean), dtype=z_mean.dtype)
            z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
            z = z.numpy()
            
            X_reconstructed = decoder.predict(z, verbose=0)
            
        # End timing
        detection_time = time.time() - start_time
        
        # Compute MSE per sample
        mse_per_sample = np.mean(np.square(X_test - X_reconstructed), axis=(1, 2))
        return mse_per_sample, detection_time
        
    except Exception as e:
        print(f"Error computing reconstruction error for {model_type}: {str(e)}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return None, None

def count_model_parameters(model, encoder_path=None, decoder_path=None, discriminator_path=None):
    """Count total parameters for single model or model ensemble"""
    total_params = 0
    
    # For single models (autoencoder, etc.), just count the main model
    if model is not None and encoder_path is None and decoder_path is None:
        total_params += model.count_params()
    
    # For VAE/GAN models, count individual components (encoder, decoder, discriminator)
    if encoder_path and os.path.exists(encoder_path):
        try:
            encoder = load_model_safe(encoder_path, "encoder")
            if encoder:
                total_params += encoder.count_params()
        except:
            pass
    
    if decoder_path and os.path.exists(decoder_path):
        try:
            decoder = load_model_safe(decoder_path, "decoder") 
            if decoder:
                total_params += decoder.count_params()
        except:
            pass
            
    if discriminator_path and os.path.exists(discriminator_path):
        try:
            discriminator = load_model_safe(discriminator_path, "discriminator")
            if discriminator:
                total_params += discriminator.count_params()
        except:
            pass
    
    return total_params


def evaluate_model(model_path, model_name, model_type, X_test, y_test, encoder_path=None, decoder_path=None, discriminator_path=None):
    """Evaluate a single model"""
    
    print(f"\n{'='*60}")
    print(f"EVALUATING: {model_name}")
    print(f"{'='*60}")
    
    model = load_model_safe(model_path, model_name)
    
    # Count total parameters
    total_params = count_model_parameters(model, encoder_path, decoder_path, discriminator_path)
    
    reconstruction_errors, detection_time = compute_reconstruction_error(X_test, model, model_type, encoder_path, decoder_path)
    
    if reconstruction_errors is None:
        return None
    
    anomaly_scores = reconstruction_errors
    
    # Calculate detection speed metrics
    samples_per_second = len(X_test) / detection_time
    milliseconds_per_sample = (detection_time * 1000) / len(X_test)
    
    precision, recall, thresholds = precision_recall_curve(y_test, anomaly_scores)
    pr_auc = auc(recall, precision)
    
    # Find optimal threshold (best F1 score)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_threshold = 0.95
    
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
        'total_parameters': total_params,
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

def plot_pr_auc_comparison(all_results):
    """Create bar plots for PR-AUC comparison"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    try:
        # Set style for better-looking plots
        plt.style.use('seaborn-v0_8')
        
        # Extract data for plotting
        model_names = [result['model_name'] for result in all_results]
        pr_auc_scores = [result['pr_auc'] * 100 for result in all_results]  # Convert to percentage
        recall_scores = [result['recall'] * 100 for result in all_results]
        f1_scores = [result['f1_score'] * 100 for result in all_results]
        
        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Define colors for each model
        colors = ['#d62728', '#2ca02c', '#ff7f0e', '#1f77b4'][:len(model_names)]
        
        # Plot 1: PR-AUC Scores
        bars1 = ax1.bar(model_names, pr_auc_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        ax1.set_title('PR-AUC Performance Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('PR-AUC Score (%)', fontsize=12)
        ax1.set_ylim(0, 100)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars1, pr_auc_scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{score:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Rotate x-axis labels for better readability
        ax1.set_xticks(range(len(model_names)))
        ax1.set_xticklabels(model_names, rotation=45, ha='right')
        
        # Plot 2: Recall Scores
        bars2 = ax2.bar(model_names, recall_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        ax2.set_title('Recall Performance Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Recall Score (%)', fontsize=12)
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars2, recall_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{score:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        ax2.set_xticks(range(len(model_names)))
        ax2.set_xticklabels(model_names, rotation=45, ha='right')
        
        # Plot 3: F1 Scores
        bars3 = ax3.bar(model_names, f1_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        ax3.set_title('F1-Score Performance Comparison', fontsize=14, fontweight='bold')
        ax3.set_ylabel('F1-Score (%)', fontsize=12)
        ax3.set_ylim(0, 100)
        ax3.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars3, f1_scores):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{score:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        ax3.set_xticks(range(len(model_names)))
        ax3.set_xticklabels(model_names, rotation=45, ha='right')
        
        # Adjust layout and save
        plt.tight_layout()
        
        # Save the plot
        output_path = 'outputs/baseline_model_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 Performance comparison plot saved to: {output_path}")
        
        # Also create a combined comparison plot
        plt.figure(figsize=(12, 8))
        
        x = np.arange(len(model_names))
        width = 0.25
        
        plt.bar(x - width, pr_auc_scores, width, label='PR-AUC', color='#d62728', alpha=0.8)
        plt.bar(x, recall_scores, width, label='Recall', color='#2ca02c', alpha=0.8)
        plt.bar(x + width, f1_scores, width, label='F1-Score', color='#ff7f0e', alpha=0.8)
        
        plt.xlabel('Models', fontsize=12)
        plt.ylabel('Performance Score (%)', fontsize=12)
        plt.title('Baseline Models Performance Comparison', fontsize=16, fontweight='bold')
        plt.xticks(x, model_names, rotation=45, ha='right')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 100)
        
        # Add value labels
        for i, (pr_auc, recall, f1) in enumerate(zip(pr_auc_scores, recall_scores, f1_scores)):
            plt.text(i - width, pr_auc + 1, f'{pr_auc:.1f}%', ha='center', va='bottom', fontsize=9)
            plt.text(i, recall + 1, f'{recall:.1f}%', ha='center', va='bottom', fontsize=9)
            plt.text(i + width, f1 + 1, f'{f1:.1f}%', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        # Save combined plot
        combined_output_path = 'outputs/baseline_model_combined_comparison.png'
        plt.savefig(combined_output_path, dpi=300, bbox_inches='tight')
        print(f"📊 Combined performance plot saved to: {combined_output_path}")
        
        plt.show()
        
    except Exception as e:
        print(f"❌ Error creating plots: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")

    return None

def plot_pr_curves(all_results, X_test, y_test):
    """Plot Precision-Recall curves for all models"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import precision_recall_curve, auc
    
    try:
        # Set style for better-looking plots
        plt.style.use('seaborn-v0_8')
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Define models to evaluate with paths
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
        
        # Plot PR curve for each model
        for model_config in models_config:
            print(f"\n📈 Computing PR curve for {model_config['model_name']}...")
            
            # Load model and compute reconstruction errors
            model = load_model_safe(model_config['model_path'], model_config['model_name'])
            if model is None:
                continue
                
            reconstruction_errors, _ = compute_reconstruction_error(
                X_test, model, model_config['model_type'], 
                model_config.get('encoder_path'), model_config.get('decoder_path')
            )
            
            if reconstruction_errors is None:
                continue
            
            # Compute precision-recall curve
            precision, recall, thresholds = precision_recall_curve(y_test, reconstruction_errors)
            pr_auc = auc(recall, precision)
            
            # Plot the curve
            plt.plot(recall, precision, 
                    color=model_config['color'], 
                    linewidth=2.5, 
                    label=f"{model_config['model_name']} (AUC = {pr_auc:.3f})")
        
        # Customize plot
        plt.xlabel('Recall', fontsize=14)
        plt.ylabel('Precision', fontsize=14)
        plt.title('Precision-Recall Curves - FDIA Detection Models', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12, loc='lower left')
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        
        plt.tight_layout()
        
        # Save the plot
        output_path = 'outputs/precision_recall_curves.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 Precision-Recall curves saved to: {output_path}")
        
        plt.show()
        
    except Exception as e:
        print(f"❌ Error creating PR curves: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")

    return None

def main():    
    print("BASELINE MODELS EVALUATION")
    print("="*80)
    
    # Load test data
    X_test, y_test = load_data()
    
    # Define models to evaluate with encoder/decoder paths for VAE models
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
            plot_pr_auc_comparison(all_results)
            plot_pr_curves(all_results, X_test, y_test)
        except Exception as e:
            print(f"⚠️  Could not create plots: {str(e)}")
    
    else:
        print("❌ No models could be evaluated successfully")
    
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
