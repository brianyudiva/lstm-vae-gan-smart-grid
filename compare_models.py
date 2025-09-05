"""
Comprehensive Model Comparison Script
Compares LSTM-VAE-GAN with all baseline models
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, auc, f1_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, accuracy_score
import pandas as pd
from datetime import datetime

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_data():
    """Load test data for evaluation"""
    data_path = '/home/brianyudiva/Documents/Project/lstm-vae-gan-smart-grid/data/sequences'
    
    X_test = np.load(os.path.join(data_path, 'X_test.npy'))
    y_test = np.load(os.path.join(data_path, 'y_test_binary.npy'))
    
    print(f"Test data loaded:")
    print(f"  X_test shape: {X_test.shape}")
    print(f"  y_test shape: {y_test.shape}")
    print(f"  Normal samples: {np.sum(y_test == 0)} ({np.mean(y_test == 0)*100:.1f}%)")
    print(f"  Anomaly samples: {np.sum(y_test == 1)} ({np.mean(y_test == 1)*100:.1f}%)")
    
    return X_test, y_test

def load_best_hyperparameters():
    """Load optimized hyperparameters"""
    best_params_path = '/home/brianyudiva/Documents/Project/lstm-vae-gan-smart-grid/best_hyperparameters.json'
    
    if os.path.exists(best_params_path):
        with open(best_params_path, 'r') as f:
            data = json.load(f)
            if 'optimized_hyperparameters' in data and 'parameters' in data['optimized_hyperparameters']:
                return data['optimized_hyperparameters']['parameters']
    
    return {'latent_dim': 32}

def calculate_reconstruction_error(model, X_test, model_type='autoencoder'):
    """Calculate reconstruction error for anomaly detection"""
    try:
        if model_type == 'vae_gan':
            # For LSTM-VAE-GAN, we need to load encoder and decoder separately
            checkpoint_dir = '/home/brianyudiva/Documents/Project/lstm-vae-gan-smart-grid/outputs/checkpoints'
            
            encoder_path = os.path.join(checkpoint_dir, 'lstm_vae_gan_encoder.h5')
            decoder_path = os.path.join(checkpoint_dir, 'lstm_vae_gan_decoder.h5')
            
            if os.path.exists(encoder_path) and os.path.exists(decoder_path):
                # Define custom sampling function for loading
                def sampling(args):
                    z_mean, z_log_var = args
                    batch = tf.shape(z_mean)[0]
                    dim = tf.shape(z_mean)[1]
                    epsilon = tf.random.normal(shape=(batch, dim))
                    return z_mean + tf.exp(0.5 * z_log_var) * epsilon
                
                # Load models with custom objects
                custom_objects = {'sampling': sampling}
                encoder = keras.models.load_model(encoder_path, custom_objects=custom_objects, compile=False)
                decoder = keras.models.load_model(decoder_path, custom_objects=custom_objects, compile=False)
                
                # Get latent representation
                encoded = encoder.predict(X_test, verbose=0)
                if isinstance(encoded, list):
                    # For VAE, take the mean (first output)
                    encoded = encoded[0]
                
                # Reconstruct
                reconstructed = decoder.predict(encoded, verbose=0)
            else:
                # Fallback: try to use the model directly (might be full VAE-GAN)
                reconstructed = model.predict(X_test, verbose=0)
                if isinstance(reconstructed, list):
                    reconstructed = reconstructed[0]
                    
        elif model_type == 'vae' or model_type == 'lstm_vae':
            # For VAE models, load with custom sampling function
            def sampling(args):
                z_mean, z_log_var = args
                batch = tf.shape(z_mean)[0]
                dim = tf.shape(z_mean)[1]
                epsilon = tf.random.normal(shape=(batch, dim))
                return z_mean + tf.exp(0.5 * z_log_var) * epsilon
            
            # Try to predict with the already loaded model, handle errors gracefully
            reconstructed = model.predict(X_test, verbose=0)
            if isinstance(reconstructed, list):
                reconstructed = reconstructed[0]  # Take the reconstruction output
        else:
            # For regular autoencoder
            reconstructed = model.predict(X_test, verbose=0)
        
        # Calculate MSE reconstruction error for each sample
        reconstruction_errors = np.mean(np.square(X_test - reconstructed), axis=(1, 2))
        return reconstruction_errors
        
    except Exception as e:
        print(f"Error calculating reconstruction error for {model_type}: {str(e)}")
        return None

def evaluate_model(model_path, model_name, X_test, y_test, model_type='autoencoder'):
    """Evaluate a single model"""
    print(f"\nEvaluating {model_name}...")
    
    try:
        # Load model
        if os.path.exists(model_path):
            # Define custom sampling function for VAE models
            def sampling(args):
                z_mean, z_log_var = args
                batch = tf.shape(z_mean)[0]
                dim = tf.shape(z_mean)[1]
                epsilon = tf.random.normal(shape=(batch, dim))
                return z_mean + tf.exp(0.5 * z_log_var) * epsilon
            
            # Load model with custom objects if it's a VAE model
            if model_type in ['vae', 'lstm_vae', 'vae_gan']:
                custom_objects = {'sampling': sampling}
                model = keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
            else:
                model = keras.models.load_model(model_path, compile=False)
                
            print(f"  Model loaded: {model.count_params():,} parameters")
        else:
            print(f"  ❌ Model file not found: {model_path}")
            return None
        
        # Calculate reconstruction errors
        errors = calculate_reconstruction_error(model, X_test, model_type)
        if errors is None:
            return None
        
        # Calculate metrics at different thresholds
        pr_scores = []
        f1_scores = []
        accuracy_scores = []
        thresholds = np.linspace(np.min(errors), np.max(errors), 100)
        
        for threshold in thresholds:
            y_pred = (errors > threshold).astype(int)
            
            if len(np.unique(y_pred)) > 1:  # Avoid division by zero
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                accuracy = accuracy_score(y_test, y_pred)
                
                pr_scores.append([precision, recall])
                f1_scores.append(f1)
                accuracy_scores.append(accuracy)
        
        # Calculate PR-AUC
        if pr_scores:
            pr_scores = np.array(pr_scores)
            precision_values = pr_scores[:, 0]
            recall_values = pr_scores[:, 1]
            pr_auc = auc(recall_values, precision_values)
        else:
            pr_auc = 0.0
        
        # Best F1 score and threshold
        if f1_scores:
            best_f1_idx = np.argmax(f1_scores)
            best_f1 = f1_scores[best_f1_idx]
            best_threshold = thresholds[best_f1_idx]
            
            # Calculate metrics at best threshold
            y_pred_best = (errors > best_threshold).astype(int)
            best_precision = precision_score(y_test, y_pred_best, zero_division=0)
            best_recall = recall_score(y_test, y_pred_best, zero_division=0)
            best_accuracy = accuracy_score(y_test, y_pred_best)
        else:
            best_f1 = best_precision = best_recall = best_accuracy = best_threshold = 0.0
        
        # Best accuracy and its threshold
        if accuracy_scores:
            best_acc_idx = np.argmax(accuracy_scores)
            max_accuracy = accuracy_scores[best_acc_idx]
            best_acc_threshold = thresholds[best_acc_idx]
        else:
            max_accuracy = best_acc_threshold = 0.0
        
        # Calculate separation ratio (how well normal/anomaly are separated)
        normal_errors = errors[y_test == 0]
        anomaly_errors = errors[y_test == 1]
        
        if len(normal_errors) > 0 and len(anomaly_errors) > 0:
            separation_ratio = np.mean(anomaly_errors) / np.mean(normal_errors)
        else:
            separation_ratio = 1.0
        
        results = {
            'model_name': model_name,
            'pr_auc': pr_auc,
            'best_f1': best_f1,
            'best_precision': best_precision,
            'best_recall': best_recall,
            'best_accuracy': best_accuracy,
            'max_accuracy': max_accuracy,
            'best_threshold': best_threshold,
            'best_acc_threshold': best_acc_threshold,
            'separation_ratio': separation_ratio,
            'mean_normal_error': np.mean(normal_errors) if len(normal_errors) > 0 else 0,
            'mean_anomaly_error': np.mean(anomaly_errors) if len(anomaly_errors) > 0 else 0,
            'reconstruction_errors': errors
        }
        
        print(f"  ✅ PR-AUC: {pr_auc:.4f}")
        print(f"  ✅ Best F1: {best_f1:.4f}")
        print(f"  ✅ Best Accuracy: {max_accuracy:.4f}")
        print(f"  ✅ Separation Ratio: {separation_ratio:.4f}")
        
        return results
        
    except Exception as e:
        print(f"  ❌ Error evaluating {model_name}: {str(e)}")
        return None

def plot_comparison_results(results, save_path):
    """Create comprehensive comparison plots"""
    
    # Filter out None results
    valid_results = [r for r in results if r is not None]
    
    if not valid_results:
        print("No valid results to plot")
        return
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Model Comparison Results', fontsize=16, fontweight='bold')
    
    # Extract data for plotting
    model_names = [r['model_name'] for r in valid_results]
    pr_aucs = [r['pr_auc'] for r in valid_results]
    f1_scores = [r['best_f1'] for r in valid_results]
    accuracies = [r['max_accuracy'] for r in valid_results]
    precisions = [r['best_precision'] for r in valid_results]
    recalls = [r['best_recall'] for r in valid_results]
    separations = [r['separation_ratio'] for r in valid_results]
    
    # 1. PR-AUC Comparison
    axes[0,0].bar(model_names, pr_aucs, color=['red', 'blue', 'green', 'orange', 'purple'][:len(model_names)])
    axes[0,0].set_title('PR-AUC Comparison')
    axes[0,0].set_ylabel('PR-AUC')
    axes[0,0].tick_params(axis='x', rotation=45)
    for i, v in enumerate(pr_aucs):
        axes[0,0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # 2. Accuracy Comparison
    axes[0,1].bar(model_names, accuracies, color=['red', 'blue', 'green', 'orange', 'purple'][:len(model_names)])
    axes[0,1].set_title('Best Accuracy Comparison')
    axes[0,1].set_ylabel('Accuracy')
    axes[0,1].tick_params(axis='x', rotation=45)
    for i, v in enumerate(accuracies):
        axes[0,1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # 3. F1 Score Comparison (moved to position 0,2)
    axes[0,2].bar(model_names, f1_scores, color=['red', 'blue', 'green', 'orange', 'purple'][:len(model_names)])
    axes[0,2].set_title('Best F1 Score Comparison')
    axes[0,2].set_ylabel('F1 Score')
    axes[0,2].tick_params(axis='x', rotation=45)
    for i, v in enumerate(f1_scores):
        axes[0,2].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # 4. Separation Ratio
    axes[1,0].bar(model_names, separations, color=['red', 'blue', 'green', 'orange', 'purple'][:len(model_names)])
    axes[1,0].set_title('Anomaly Separation Ratio')
    axes[1,0].set_ylabel('Separation Ratio')
    axes[1,0].tick_params(axis='x', rotation=45)
    for i, v in enumerate(separations):
        axes[1,0].text(i, v + 0.1, f'{v:.2f}', ha='center', va='bottom')
    
    # 5. Precision vs Recall (moved from top row)
    axes[1,1].scatter(recalls, precisions, s=100, c=['red', 'blue', 'green', 'orange', 'purple'][:len(model_names)])
    for i, name in enumerate(model_names):
        axes[1,1].annotate(name, (recalls[i], precisions[i]), xytext=(5, 5), 
                          textcoords='offset points', fontsize=8)
    axes[1,1].set_xlabel('Recall')
    axes[1,1].set_ylabel('Precision')
    axes[1,1].set_title('Precision vs Recall')
    axes[1,1].grid(True, alpha=0.3)
    
    # 6. Performance Summary Table
    axes[1,2].axis('tight')
    axes[1,2].axis('off')
    
    # Create summary table
    table_data = []
    for r in valid_results:
        table_data.append([
            r['model_name'],
            f"{r['max_accuracy']:.3f}",
            f"{r['pr_auc']:.3f}",
            f"{r['best_f1']:.3f}",
            f"{r['separation_ratio']:.2f}"
        ])
    
    table = axes[1,2].table(cellText=table_data,
                           colLabels=['Model', 'Accuracy', 'PR-AUC', 'F1', 'Sep.Ratio'],
                           cellLoc='center',
                           loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    axes[1,2].set_title('Performance Summary')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Comparison plots saved to: {save_path}")

def main():
    """Main comparison function"""
    
    print("="*80)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("="*80)
    
    # Load test data
    X_test, y_test = load_data()
    params = load_best_hyperparameters()
    
    # Define models to compare
    models_to_compare = [
        {
            'name': 'LSTM-VAE-GAN (Proposed)',
            'path': '/home/brianyudiva/Documents/Project/lstm-vae-gan-smart-grid/outputs/checkpoints/lstm_vae_gan_encoder.h5',
            'type': 'vae_gan'
        },
        {
            'name': 'LSTM Autoencoder',
            'path': '/home/brianyudiva/Documents/Project/lstm-vae-gan-smart-grid/outputs/checkpoints/lstm_autoencoder_full.h5',
            'type': 'autoencoder'
        },
        {
            'name': 'VAE-GAN',
            'path': '/home/brianyudiva/Documents/Project/lstm-vae-gan-smart-grid/outputs/checkpoints/vae_gan_full.h5',
            'type': 'vae'
        },
        {
            'name': 'LSTM-GAN',
            'path': '/home/brianyudiva/Documents/Project/lstm-vae-gan-smart-grid/outputs/checkpoints/lstm_gan_generator.h5',
            'type': 'autoencoder'
        },
        {
            'name': 'LSTM-VAE',
            'path': '/home/brianyudiva/Documents/Project/lstm-vae-gan-smart-grid/outputs/checkpoints/lstm_vae_full.h5',
            'type': 'lstm_vae'
        }
    ]
    
    # Evaluate all models
    results = []
    for model_info in models_to_compare:
        result = evaluate_model(
            model_info['path'], 
            model_info['name'], 
            X_test, 
            y_test, 
            model_info['type']
        )
        results.append(result)
    
    # Filter valid results and create summary
    valid_results = [r for r in results if r is not None]
    
    if valid_results:
        print("\n" + "="*80)
        print("COMPARISON SUMMARY")
        print("="*80)
        
        # Sort by Accuracy (primary metric)
        valid_results.sort(key=lambda x: x['max_accuracy'], reverse=True)
        
        print(f"{'Rank':<4} {'Model':<25} {'Acc':<8} {'PR-AUC':<8} {'F1':<8} {'Prec':<8} {'Rec':<8} {'Sep.Ratio':<10}")
        print("-" * 90)
        
        for i, result in enumerate(valid_results, 1):
            print(f"{i:<4} {result['model_name']:<25} "
                  f"{result['max_accuracy']:<8.4f} {result['pr_auc']:<8.4f} "
                  f"{result['best_f1']:<8.4f} {result['best_precision']:<8.4f} "
                  f"{result['best_recall']:<8.4f} {result['separation_ratio']:<10.2f}")
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f'/home/brianyudiva/Documents/Project/lstm-vae-gan-smart-grid/outputs/model_comparison_{timestamp}.json'
        
        # Prepare results for JSON (remove numpy arrays)
        json_results = []
        for r in valid_results:
            json_result = {k: v for k, v in r.items() if k != 'reconstruction_errors'}
            json_result['reconstruction_errors_stats'] = {
                'mean': float(np.mean(r['reconstruction_errors'])),
                'std': float(np.std(r['reconstruction_errors'])),
                'min': float(np.min(r['reconstruction_errors'])),
                'max': float(np.max(r['reconstruction_errors']))
            }
            json_results.append(json_result)
        
        with open(results_file, 'w') as f:
            json.dump({
                'comparison_date': timestamp,
                'test_data_size': len(X_test),
                'anomaly_ratio': float(np.mean(y_test)),
                'results': json_results
            }, f, indent=2)
        
        print(f"\nDetailed results saved to: {results_file}")
        
        # Create comparison plots
        plot_path = f'/home/brianyudiva/Documents/Project/lstm-vae-gan-smart-grid/outputs/model_comparison_{timestamp}.png'
        plot_comparison_results(valid_results, plot_path)
        
        # Performance insights
        best_model = valid_results[0]
        print(f"\n🏆 BEST PERFORMING MODEL: {best_model['model_name']}")
        print(f"   PR-AUC: {best_model['pr_auc']:.4f}")
        print(f"   F1 Score: {best_model['best_f1']:.4f}")
        print(f"   Accuracy: {best_model['max_accuracy']:.4f}")
        print(f"   Separation Ratio: {best_model['separation_ratio']:.2f}")
        
        if len(valid_results) > 1:
            improvement = (best_model['max_accuracy'] - valid_results[1]['max_accuracy']) / valid_results[1]['max_accuracy'] * 100
            print(f"   Improvement over 2nd best: {improvement:.1f}%")
    
    else:
        print("❌ No models could be evaluated successfully")

if __name__ == "__main__":
    main()
