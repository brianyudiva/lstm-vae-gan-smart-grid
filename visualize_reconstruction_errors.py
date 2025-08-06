#!/usr/bin/env python3
"""
Visualization script for reconstruction error histograms
Plots histograms of reconstruction error for normal vs. FDIA samples
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_data():
    """Load the preprocessed data and labels"""
    sequence_path = "data/sequences"
    
    # Load test data
    X_test = np.load(f"{sequence_path}/X_test.npy")
    y_test = np.load(f"{sequence_path}/y_test_binary.npy")
    
    # Load training data for normal samples
    X_train = np.load(f"{sequence_path}/X_train.npy")
    y_train = np.load(f"{sequence_path}/y_train_binary.npy")
    X_train_normal = X_train[y_train == 0]
    
    print(f"Test data loaded: {X_test.shape}")
    print(f"Test labels - Normal: {np.sum(y_test == 0)}, FDIA: {np.sum(y_test == 1)}")
    print(f"Training normal samples: {X_train_normal.shape}")
    
    return X_test, y_test, X_train_normal

def load_trained_models(input_shape, latent_dim=8):
    """Recreate models and load weights to avoid custom function loading issues"""
    from models.lstm_vae_gan import select_architecture
    
    output_path = "outputs/checkpoints"
    model_prefix = "lstm_vae_gan_pure_anomaly"
    
    encoder_path = f"{output_path}/{model_prefix}_anomaly_best_encoder.h5"
    decoder_path = f"{output_path}/{model_prefix}_anomaly_best_decoder.h5"
    
    if not os.path.exists(encoder_path) or not os.path.exists(decoder_path):
        raise FileNotFoundError(f"Trained models not found. Please run training first.")
    
    # Recreate the models with the same architecture
    # We'll try with different latent dimensions to find the right one
    for test_latent_dim in [8, 12, 16, 32]:
        try:
            print(f"Trying to load models with latent_dim={test_latent_dim}...")
            
            # Create fresh models
            encoder, decoder, discriminator, arch_info = select_architecture(
                normal_samples_count=1000,  # Dummy value
                input_shape=input_shape,
                latent_dim=test_latent_dim,
                force_architecture=None
            )
            
            # Try to load weights
            encoder.load_weights(encoder_path)
            decoder.load_weights(decoder_path)
            
            print(f"Successfully loaded models with latent_dim={test_latent_dim}")
            print(f"Architecture: {arch_info['name']}")
            return encoder, decoder
            
        except Exception as e:
            print(f"Failed with latent_dim={test_latent_dim}: {str(e)[:100]}...")
            continue
    
    # If all attempts fail, try a different approach
    print("Attempting alternative loading method...")
    try:
        # Simple reconstruction without custom functions
        encoder = tf.keras.models.load_model(encoder_path, compile=False, custom_objects={'sampling': lambda x: x})
        decoder = tf.keras.models.load_model(decoder_path, compile=False)
        return encoder, decoder
    except Exception as e:
        print(f"Alternative method also failed: {e}")
        raise RuntimeError("Could not load trained models. Please retrain the models.")

def compute_reconstruction_errors(encoder, decoder, X_data):
    """Compute reconstruction errors for the given data"""
    print("Computing reconstruction errors...")
    
    # Forward pass through encoder
    if hasattr(encoder, 'predict'):
        # If encoder returns multiple outputs (z_mean, z_log_var, z)
        encoder_output = encoder.predict(X_data, batch_size=32, verbose=0)
        if isinstance(encoder_output, list) and len(encoder_output) >= 3:
            z = encoder_output[2]  # Use the sampled z
        else:
            z = encoder_output
    else:
        z = encoder(X_data, training=False)
        if isinstance(z, list) and len(z) >= 3:
            z = z[2]  # Use the sampled z
    
    # Forward pass through decoder
    if hasattr(decoder, 'predict'):
        reconstructed = decoder.predict(z, batch_size=32, verbose=0)
    else:
        reconstructed = decoder(z, training=False)
    
    # Compute reconstruction errors (MSE per sample)
    errors = tf.reduce_mean(tf.square(X_data - reconstructed), axis=[1, 2])
    return errors.numpy()

def create_histogram_visualization(normal_errors, fdia_errors, save_path="outputs/reconstruction_error_histograms.png"):
    """Create comprehensive histogram visualization"""
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Reconstruction Error Analysis: Normal vs. FDIA Samples', fontsize=16, fontweight='bold')
    
    # Plot 1: Overlapping histograms
    axes[0, 0].hist(normal_errors, bins=50, alpha=0.7, label='Normal', color='blue', density=True)
    axes[0, 0].hist(fdia_errors, bins=50, alpha=0.7, label='FDIA', color='red', density=True)
    axes[0, 0].set_xlabel('Reconstruction Error')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Overlapping Histograms')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Side-by-side histograms
    x = np.arange(2)
    mean_errors = [np.mean(normal_errors), np.mean(fdia_errors)]
    std_errors = [np.std(normal_errors), np.std(fdia_errors)]
    colors = ['blue', 'red']
    labels = ['Normal', 'FDIA']
    
    bars = axes[0, 1].bar(x, mean_errors, yerr=std_errors, capsize=5, color=colors, alpha=0.7)
    axes[0, 1].set_xlabel('Sample Type')
    axes[0, 1].set_ylabel('Mean Reconstruction Error')
    axes[0, 1].set_title('Mean Reconstruction Error Comparison')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(labels)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, mean_val, std_val) in enumerate(zip(bars, mean_errors, std_errors)):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + std_val + 0.0001,
                       f'{mean_val:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Box plots
    data_for_box = [normal_errors, fdia_errors]
    box_plot = axes[1, 0].boxplot(data_for_box, labels=['Normal', 'FDIA'], patch_artist=True)
    box_plot['boxes'][0].set_facecolor('blue')
    box_plot['boxes'][1].set_facecolor('red')
    for patch in box_plot['boxes']:
        patch.set_alpha(0.7)
    
    axes[1, 0].set_ylabel('Reconstruction Error')
    axes[1, 0].set_title('Box Plot Comparison')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Cumulative Distribution Function
    sorted_normal = np.sort(normal_errors)
    sorted_fdia = np.sort(fdia_errors)
    y_normal = np.arange(1, len(sorted_normal) + 1) / len(sorted_normal)
    y_fdia = np.arange(1, len(sorted_fdia) + 1) / len(sorted_fdia)
    
    axes[1, 1].plot(sorted_normal, y_normal, label='Normal', color='blue', linewidth=2)
    axes[1, 1].plot(sorted_fdia, y_fdia, label='FDIA', color='red', linewidth=2)
    axes[1, 1].set_xlabel('Reconstruction Error')
    axes[1, 1].set_ylabel('Cumulative Probability')
    axes[1, 1].set_title('Cumulative Distribution Function')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Histogram visualization saved to {save_path}")

def print_statistical_analysis(normal_errors, fdia_errors):
    """Print detailed statistical analysis"""
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS OF RECONSTRUCTION ERRORS")
    print("="*60)
    
    # Basic statistics
    print(f"\nNORMAL SAMPLES ({len(normal_errors)} samples):")
    print(f"  Mean:     {np.mean(normal_errors):.6f}")
    print(f"  Std:      {np.std(normal_errors):.6f}")
    print(f"  Min:      {np.min(normal_errors):.6f}")
    print(f"  Max:      {np.max(normal_errors):.6f}")
    print(f"  Median:   {np.median(normal_errors):.6f}")
    print(f"  Q1:       {np.percentile(normal_errors, 25):.6f}")
    print(f"  Q3:       {np.percentile(normal_errors, 75):.6f}")
    
    print(f"\nFDIA SAMPLES ({len(fdia_errors)} samples):")
    print(f"  Mean:     {np.mean(fdia_errors):.6f}")
    print(f"  Std:      {np.std(fdia_errors):.6f}")
    print(f"  Min:      {np.min(fdia_errors):.6f}")
    print(f"  Max:      {np.max(fdia_errors):.6f}")
    print(f"  Median:   {np.median(fdia_errors):.6f}")
    print(f"  Q1:       {np.percentile(fdia_errors, 25):.6f}")
    print(f"  Q3:       {np.percentile(fdia_errors, 75):.6f}")
    
    # Separation metrics
    separation_ratio = np.mean(fdia_errors) / np.mean(normal_errors)
    print(f"\nSEPARATION ANALYSIS:")
    print(f"  Mean Separation Ratio: {separation_ratio:.3f}x")
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.std(normal_errors)**2 + np.std(fdia_errors)**2) / 2)
    effect_size = (np.mean(fdia_errors) - np.mean(normal_errors)) / pooled_std
    print(f"  Effect Size (Cohen's d): {effect_size:.3f}")
    
    if effect_size < 0.2:
        effect_interpretation = "negligible"
    elif effect_size < 0.5:
        effect_interpretation = "small"
    elif effect_size < 0.8:
        effect_interpretation = "medium"
    else:
        effect_interpretation = "large"
    print(f"  Effect Size Interpretation: {effect_interpretation}")
    
    # Statistical significance test
    try:
        t_stat, p_value = stats.ttest_ind(fdia_errors, normal_errors, equal_var=False)
        print(f"\nSTATISTICAL SIGNIFICANCE:")
        print(f"  T-statistic: {t_stat:.4f}")
        print(f"  P-value: {p_value:.2e}")
        print(f"  Significant at α=0.001: {'Yes' if p_value < 0.001 else 'No'}")
        print(f"  Significant at α=0.05: {'Yes' if p_value < 0.05 else 'No'}")
    except Exception as e:
        print(f"  Could not compute t-test: {e}")
    
    # Kolmogorov-Smirnov test
    try:
        ks_stat, ks_p_value = stats.ks_2samp(normal_errors, fdia_errors)
        print(f"\nDISTRIBUTION COMPARISON (KS Test):")
        print(f"  KS statistic: {ks_stat:.4f}")
        print(f"  P-value: {ks_p_value:.2e}")
        print(f"  Distributions different: {'Yes' if ks_p_value < 0.001 else 'No'}")
    except Exception as e:
        print(f"  Could not compute KS test: {e}")
    
    # Overlap analysis
    normal_max = np.max(normal_errors)
    fdia_min = np.min(fdia_errors)
    if fdia_min > normal_max:
        print(f"\nOVERLAP ANALYSIS:")
        print(f"  Perfect separation: Yes (no overlap)")
        print(f"  Gap: {fdia_min - normal_max:.6f}")
    else:
        # Calculate overlap percentage
        combined = np.concatenate([normal_errors, fdia_errors])
        overlap_threshold = np.linspace(np.min(combined), np.max(combined), 1000)
        overlap_area = 0
        for thresh in overlap_threshold[:-1]:
            normal_above = np.mean(normal_errors > thresh)
            fdia_below = np.mean(fdia_errors <= thresh)
            overlap_area += min(normal_above, 1 - fdia_below)
        overlap_percentage = (overlap_area / len(overlap_threshold)) * 100
        print(f"\nOVERLAP ANALYSIS:")
        print(f"  Perfect separation: No")
        print(f"  Estimated overlap: {overlap_percentage:.1f}%")
    
    # Classification potential
    print(f"\nCLASSIFICATION POTENTIAL:")
    if separation_ratio > 3.0:
        print(f"  Assessment: EXCELLENT - Strong separation for anomaly detection")
    elif separation_ratio > 2.0:
        print(f"  Assessment: GOOD - Clear separation, suitable for anomaly detection")
    elif separation_ratio > 1.5:
        print(f"  Assessment: MODERATE - Some separation, may work with proper threshold")
    elif separation_ratio > 1.2:
        print(f"  Assessment: WEAK - Limited separation, challenging for anomaly detection")
    else:
        print(f"  Assessment: POOR - No meaningful separation, model needs improvement")

def suggest_optimal_thresholds(normal_errors, fdia_errors):
    """Suggest optimal thresholds for anomaly detection"""
    print(f"\nOPTIMAL THRESHOLD SUGGESTIONS:")
    print("-" * 40)
    
    # Percentile-based thresholds
    percentiles = [90, 95, 97, 99, 99.5]
    print(f"Percentile-based thresholds (from normal data):")
    for p in percentiles:
        threshold = np.percentile(normal_errors, p)
        normal_above = np.mean(normal_errors > threshold)
        fdia_above = np.mean(fdia_errors > threshold)
        print(f"  {p:4.1f}th percentile: {threshold:.6f}")
        print(f"    False Positive Rate: {normal_above*100:.2f}%")
        print(f"    True Positive Rate:  {fdia_above*100:.2f}%")
        if normal_above > 0:
            precision = fdia_above * len(fdia_errors) / (fdia_above * len(fdia_errors) + normal_above * len(normal_errors))
            print(f"    Precision: {precision:.3f}")
        print()

def main():
    """Main function to run the visualization"""
    print("Loading data and models...")
    
    # Load data
    X_test, y_test, X_train_normal = load_data()
    
    # Load trained models
    input_shape = (X_test.shape[1], X_test.shape[2])  # (12, 62)
    encoder, decoder = load_trained_models(input_shape)
    
    # Compute reconstruction errors for test data
    test_errors = compute_reconstruction_errors(encoder, decoder, X_test)
    
    # Separate errors by label
    normal_errors = test_errors[y_test == 0]
    fdia_errors = test_errors[y_test == 1]
    
    print(f"\nReconstructed {len(test_errors)} test samples")
    print(f"Normal samples: {len(normal_errors)}")
    print(f"FDIA samples: {len(fdia_errors)}")
    
    # Create visualization
    create_histogram_visualization(normal_errors, fdia_errors)
    
    # Print statistical analysis
    print_statistical_analysis(normal_errors, fdia_errors)
    
    # Suggest optimal thresholds
    suggest_optimal_thresholds(normal_errors, fdia_errors)
    
    print(f"\n{'='*60}")
    print("VISUALIZATION COMPLETE!")
    print("Check 'outputs/reconstruction_error_histograms.png' for the detailed plots.")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
