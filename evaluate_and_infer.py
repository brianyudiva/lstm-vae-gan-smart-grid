import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, roc_curve
import seaborn as sns

sequence_path = "data/sequences"

# === CUSTOM OBJECTS FOR LOADING ===
# Define custom sampling function for loading VAE models
def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.random.normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def compute_reconstruction_errors(encoder, decoder, X_data):
    """Compute reconstruction errors for given data"""
    try:
        # Handle both VAE (3 outputs) and regular encoder (1 output)
        encoder_output = encoder(X_data)
        if isinstance(encoder_output, list) and len(encoder_output) == 3:
            # VAE case: z_mean, z_log_var, z
            z_mean, z_log_var, z = encoder_output
            reconstructed = decoder(z)
        else:
            # Regular autoencoder case
            reconstructed = decoder(encoder_output)
        
        # Calculate reconstruction errors
        recon_errors = tf.reduce_mean(tf.square(X_data - reconstructed), axis=[1, 2])
        return recon_errors.numpy()
    
    except Exception as e:
        print(f"Error in reconstruction: {e}")
        return None

# === LOAD MODELS ===
try:
    # Try loading with custom objects
    custom_objects = {'sampling': sampling}
    encoder = tf.keras.models.load_model("outputs/checkpoints/compact_encoder_best.h5", custom_objects=custom_objects)
    decoder = tf.keras.models.load_model("outputs/checkpoints/compact_decoder_best.h5", custom_objects=custom_objects)
    print("✅ Models loaded successfully")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    print("💡 Tip: Make sure models were saved correctly and custom layers are defined")
    exit()

# === LOAD DATA ===
X_train = np.load(f"{sequence_path}/X_train.npy")
X_test = np.load(f"{sequence_path}/X_test.npy")

# Load labels
try:
    y_train = np.load(f"{sequence_path}/y_train_binary.npy")
    y_test = np.load(f"{sequence_path}/y_test_binary.npy")
    print(f"Labels loaded - Train FDIA: {np.sum(y_train)}/{len(y_train)}, Test FDIA: {np.sum(y_test)}/{len(y_test)}")
    
    # Get normal training data for threshold calculation
    X_train_normal = X_train[y_train == 0]
    print(f"Normal training samples: {len(X_train_normal)}")
    
except Exception as e:
    print(f"No labels found: {e}")
    X_train_normal = X_train
    y_train, y_test = None, None

# Add this right after loading the data in evaluate_and_infer.py

print("\n" + "="*60)
print("🔬 DETAILED DIAGNOSTIC ANALYSIS")
print("="*60)

# 1. Verify data separation
print(f"📊 DATA VERIFICATION:")
print(f"X_train shape: {X_train.shape}")
print(f"X_train_normal shape: {X_train_normal.shape}")
print(f"Excluded FDIA samples: {len(X_train) - len(X_train_normal)}")
print(f"Training data reduction: {(1 - len(X_train_normal)/len(X_train))*100:.1f}%")

# 2. Check actual error separation
def detailed_error_analysis():
    """Analyze reconstruction errors in detail"""
    print(f"\n📈 RECONSTRUCTION ERROR ANALYSIS:")
    
    # Compute errors
    test_errors = compute_reconstruction_errors(encoder, decoder, X_test)
    train_errors = compute_reconstruction_errors(encoder, decoder, X_train_normal)
    
    if test_errors is None or train_errors is None:
        print("❌ Could not compute reconstruction errors")
        return None, None, None, None
    
    # Separate test errors by class
    normal_test_errors = test_errors[y_test == 0]
    fdia_test_errors = test_errors[y_test == 1]
    
    print(f"Train (Normal only): {np.mean(train_errors):.6f} ± {np.std(train_errors):.6f}")
    print(f"Test Normal: {np.mean(normal_test_errors):.6f} ± {np.std(normal_test_errors):.6f}")
    print(f"Test FDIA: {np.mean(fdia_test_errors):.6f} ± {np.std(fdia_test_errors):.6f}")
    
    # Calculate separation metrics
    separation_ratio = np.mean(fdia_test_errors) / np.mean(normal_test_errors)
    print(f"🎯 FDIA/Normal separation ratio: {separation_ratio:.3f}x")
    
    # Statistical significance
    from scipy import stats
    t_stat, p_value = stats.ttest_ind(normal_test_errors, fdia_test_errors)
    print(f"📊 T-test p-value: {p_value:.6f} ({'Significant' if p_value < 0.05 else 'Not significant'})")
    
    # Overlap analysis
    normal_95th = np.percentile(normal_test_errors, 95)
    fdia_below_95th = np.sum(fdia_test_errors < normal_95th)
    overlap_percentage = fdia_below_95th / len(fdia_test_errors) * 100
    
    print(f"🔄 FDIA samples below normal 95th percentile: {fdia_below_95th}/{len(fdia_test_errors)} ({overlap_percentage:.1f}%)")
    
    if separation_ratio < 2.0:
        print("🚨 WARNING: Poor separation! Model likely trained on mixed data or too complex")
    if overlap_percentage > 50:
        print("🚨 WARNING: High overlap! Model cannot distinguish attacks from normal data")
    
    return train_errors, normal_test_errors, fdia_test_errors, test_errors

# 3. Model complexity analysis
def analyze_model_complexity():
    """Check if model is too complex"""
    print(f"\n🏗️ MODEL COMPLEXITY ANALYSIS:")

    encoder_params = sum([np.prod(v.shape) for v in encoder.trainable_weights])
    decoder_params = sum([np.prod(v.shape) for v in decoder.trainable_weights])
    total_params = encoder_params + decoder_params
    
    print(f"Encoder parameters: {encoder_params:,}")
    print(f"Decoder parameters: {decoder_params:,}")
    print(f"Total parameters: {total_params:,}")
    
    # Data to parameter ratio
    data_param_ratio = len(X_train_normal) / total_params
    print(f"Data/Parameter ratio: {data_param_ratio:.2f}")
    
    if total_params > 100000:
        print("⚠️ WARNING: Very large model - may overfit")
    if data_param_ratio < 10:
        print("⚠️ WARNING: More parameters than data - likely overfitting")

# 4. Training verification
def check_training_contamination():
    """Check if training data contains FDIA patterns"""
    print(f"\n🔍 TRAINING DATA CONTAMINATION CHECK:")
    
    # This is the key check - verify that training used ONLY normal data
    print(f"Original training data: {len(X_train)} samples")
    print(f"FDIA in original training: {np.sum(y_train == 1)} samples")
    print(f"Used for training: {len(X_train_normal)} samples")
    
    # Check if there's any mismatch
    expected_normal = np.sum(y_train == 0)
    if len(X_train_normal) != expected_normal:
        print(f"🚨 MISMATCH: Expected {expected_normal} normal samples, got {len(X_train_normal)}")
        print("🔧 FIX: Retrain model using ONLY normal data")
    else:
        print("✅ Training data selection looks correct")
    
    # Additional check: see if current model file was trained correctly
    try:
        # Check model modification time vs current time
        import os
        model_path = "outputs/checkpoints/best_encoder.h5"
        if os.path.exists(model_path):
            mod_time = os.path.getmtime(model_path)
            import time
            current_time = time.time()
            hours_ago = (current_time - mod_time) / 3600
            print(f"Model was saved {hours_ago:.1f} hours ago")
        
    except:
        pass

# Run all diagnostics
train_errs, normal_test_errs, fdia_test_errs, all_test_errs = detailed_error_analysis()
analyze_model_complexity()
check_training_contamination()

print("\n" + "="*60)
print("🎯 RECOMMENDATIONS")
print("="*60)

if train_errs is not None:
    separation = np.mean(fdia_test_errs) / np.mean(normal_test_errs)
    
    if separation < 2.0:
        print("❌ CRITICAL: Model cannot distinguish FDIA from normal data")
        print("🔧 SOLUTION 1: Retrain with simplified architecture")
        print("🔧 SOLUTION 2: Use only normal data for training (verify this is actually happening)")
        print("🔧 SOLUTION 3: Increase reconstruction loss weight to 1000+")
    elif separation < 3.0:
        print("⚠️ MODERATE: Some separation but not optimal")
        print("🔧 SOLUTION: Fine-tune hyperparameters")
    else:
        print("✅ GOOD: Model shows proper separation")

def comprehensive_evaluation(encoder, decoder, X_test, y_test, X_train_normal):
    """Comprehensive evaluation with multiple metrics"""
    
    # Get reconstruction errors
    print("Computing reconstruction errors...")
    test_errors = compute_reconstruction_errors(encoder, decoder, X_test)
    train_errors = compute_reconstruction_errors(encoder, decoder, X_train_normal)
    
    if test_errors is None or train_errors is None:
        print("❌ Failed to compute reconstruction errors")
        return
    
    print(f"Test reconstruction errors - Mean: {np.mean(test_errors):.4f}, Std: {np.std(test_errors):.4f}")
    print(f"Train reconstruction errors - Mean: {np.mean(train_errors):.4f}, Std: {np.std(train_errors):.4f}")
    
    # Calculate AUC-ROC
    if y_test is not None:
        auc_roc = roc_auc_score(y_test, test_errors)
        print(f"\n📊 AUC-ROC Score: {auc_roc:.4f}")
        
        # Plot ROC curve
        fpr, tpr, roc_thresholds = roc_curve(y_test, test_errors)
        
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_roc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
    
    # Test different threshold strategies
    threshold_strategies = [
        ("Train 90th percentile", np.percentile(train_errors, 90)),
        ("Train 95th percentile", np.percentile(train_errors, 95)),
        ("Train 99th percentile", np.percentile(train_errors, 99)),
        ("Train Mean + 2*Std", np.mean(train_errors) + 2*np.std(train_errors)),
        ("Train Mean + 3*Std", np.mean(train_errors) + 3*np.std(train_errors)),
    ]
    
    if y_test is not None:
        # Find optimal threshold using Youden's index
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = roc_thresholds[optimal_idx]
        threshold_strategies.append(("Optimal (Youden's)", optimal_threshold))
    
    print("\n" + "="*60)
    print("📋 THRESHOLD EVALUATION RESULTS")
    print("="*60)
    
    best_f1 = 0
    best_result = None
    results = []
    
    for name, threshold in threshold_strategies:
        predictions = (test_errors > threshold).astype(int)
        n_predicted_anomalies = np.sum(predictions)
        
        result = {
            'name': name,
            'threshold': threshold,
            'predictions': predictions,
            'n_predicted': n_predicted_anomalies
        }
        
        if y_test is not None and len(np.unique(y_test)) > 1:
            # Calculate detailed metrics
            tn = np.sum((predictions == 0) & (y_test == 0))
            fp = np.sum((predictions == 1) & (y_test == 0))
            fn = np.sum((predictions == 0) & (y_test == 1))
            tp = np.sum((predictions == 1) & (y_test == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / len(y_test)
            
            result.update({
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy,
                'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
            })
            
            print(f"\n🎯 {name}: {threshold:.4f}")
            print(f"   Predicted anomalies: {n_predicted_anomalies}/{len(predictions)} ({n_predicted_anomalies/len(predictions)*100:.1f}%)")
            print(f"   Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f} | Accuracy: {accuracy:.3f}")
            print(f"   TP: {tp} | FP: {fp} | TN: {tn} | FN: {fn}")
            
            if f1 > best_f1:
                best_f1 = f1
                best_result = result
        else:
            print(f"\n🎯 {name}: {threshold:.4f}")
            print(f"   Predicted anomalies: {n_predicted_anomalies}/{len(predictions)} ({n_predicted_anomalies/len(predictions)*100:.1f}%)")
        
        results.append(result)
    
    if best_result and y_test is not None:
        print(f"\n🏆 BEST THRESHOLD: {best_result['name']}")
        print(f"   Threshold: {best_result['threshold']:.4f}")
        print(f"   F1-Score: {best_result['f1']:.3f}")
        
        # Plot confusion matrix for best threshold
        if y_test is not None:
            plt.subplot(1, 2, 2)
            cm = confusion_matrix(y_test, best_result['predictions'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix\n{best_result["name"]}')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
    
    if y_test is not None:
        plt.tight_layout()
        plt.savefig('outputs/evaluation_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Plot reconstruction error distributions
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(train_errors, bins=50, alpha=0.7, label='Training (Normal)', color='blue')
    if y_test is not None:
        normal_test_errors = test_errors[y_test == 0]
        attack_test_errors = test_errors[y_test == 1]
        plt.hist(normal_test_errors, bins=50, alpha=0.7, label='Test (Normal)', color='green')
        plt.hist(attack_test_errors, bins=50, alpha=0.7, label='Test (FDIA)', color='red')
    else:
        plt.hist(test_errors, bins=50, alpha=0.7, label='Test', color='orange')
    
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.title('Reconstruction Error Distributions')
    plt.legend()
    plt.yscale('log')
    
    plt.subplot(1, 2, 2)
    if y_test is not None:
        plt.boxplot([train_errors, normal_test_errors, attack_test_errors], 
                   labels=['Train (Normal)', 'Test (Normal)', 'Test (FDIA)'])
    else:
        plt.boxplot([train_errors, test_errors], labels=['Train', 'Test'])
    plt.ylabel('Reconstruction Error')
    plt.title('Reconstruction Error Box Plot')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('outputs/reconstruction_errors.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results

# === RUN COMPREHENSIVE EVALUATION ===
if y_test is not None:
    results = comprehensive_evaluation(encoder, decoder, X_test, y_test, X_train_normal)
else:
    print("⚠️  No labels available - running basic evaluation")
    test_errors = compute_reconstruction_errors(encoder, decoder, X_test)
    train_errors = compute_reconstruction_errors(encoder, decoder, X_train_normal)
    
    if test_errors is not None and train_errors is not None:
        threshold_95 = np.percentile(train_errors, 95)
        predictions = (test_errors > threshold_95).astype(int)
        print(f"Anomaly threshold (95th percentile): {threshold_95:.4f}")
        print(f"Detected anomalies: {np.sum(predictions)}/{len(predictions)} ({np.sum(predictions)/len(predictions)*100:.1f}%)")

print("\n✅ Evaluation completed!")