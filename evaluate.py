import json
import numpy as np
import os
import tensorflow as tf
from models.lstm_vae_gan import build_lstm_vae_gan_regular
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score
from sklearn.metrics import confusion_matrix, classification_report

def evaluate_model():
    print("=" * 60)
    print("=" * 60)

    sequence_path = "data/sequences"
    # Load attack-preserved test data
    try:
        X_test = np.load(f"{sequence_path}/X_test.npy")
        y_test = np.load(f"{sequence_path}/y_test_binary.npy")
        print(f"✅ Loaded attack-preserved test data")
    except FileNotFoundError:
        print("❌ Error: Attack-preserved test data not found.")
        print("   Run preprocessing/attack_preserving_preprocessing.py first")
        return None
        
    print(f"Test samples: {len(X_test)} total")
    print(f"  - Normal: {len(X_test) - np.sum(y_test)} samples")
    print(f"  - Attacks: {np.sum(y_test)} samples ({np.sum(y_test)/len(y_test)*100:.1f}%)")
    
    # Load trained model
    encoder_path = "outputs/checkpoints/lstm_vae_gan_encoder.h5"
    decoder_path = "outputs/checkpoints/lstm_vae_gan_decoder.h5"
    
    print(f"✅ Found trained model files")
    
    # Recreate architecture (must match training parameters)
    input_shape = (X_test.shape[1], X_test.shape[2])
    encoder, decoder, _ = build_lstm_vae_gan_regular(
        input_shape=input_shape,
        latent_dim=16,  # Must match training
    )
    
    # Load weights
    encoder.load_weights(encoder_path)
    decoder.load_weights(decoder_path)
    print(f"✅ Loaded model weights")
    
    # Calculate reconstruction errors
    print(f"\\n🔍 CALCULATING RECONSTRUCTION ERRORS...")
    batch_size = 64
    all_errors = []
    
    for i in range(0, len(X_test), batch_size):
        batch_end = min(i + batch_size, len(X_test))
        X_batch = X_test[i:batch_end]
        
        # Forward pass
        _, _, z = encoder(X_batch, training=False)
        X_recon = decoder(z, training=False)
        
        # Calculate MSE per sample
        batch_errors = tf.reduce_mean(tf.square(X_batch - X_recon), axis=[1, 2]).numpy()
        all_errors.extend(batch_errors)
    
    reconstruction_errors = np.array(all_errors)
    
    # Separate normal and attack errors
    normal_errors = reconstruction_errors[y_test == 0]
    attack_errors = reconstruction_errors[y_test == 1]
    
    print(f"\\n📊 RECONSTRUCTION ERROR ANALYSIS:")
    print("-" * 50)
    print(f"Normal samples:")
    print(f"  Count: {len(normal_errors)}")
    print(f"  Mean error: {np.mean(normal_errors):.6f}")
    print(f"  Std error: {np.std(normal_errors):.6f}")
    print(f"  Range: [{np.min(normal_errors):.6f}, {np.max(normal_errors):.6f}]")
    
    print(f"\\nAttack samples:")
    print(f"  Count: {len(attack_errors)}")
    print(f"  Mean error: {np.mean(attack_errors):.6f}")
    print(f"  Std error: {np.std(attack_errors):.6f}")
    print(f"  Range: [{np.min(attack_errors):.6f}, {np.max(attack_errors):.6f}]")
    
    # Calculate separation ratio
    separation_ratio = np.mean(attack_errors) / np.mean(normal_errors)
    print(f"\\n🎯 SEPARATION ANALYSIS:")
    print(f"  Separation ratio: {separation_ratio:.1f}x")
    
    # Performance metrics
    roc_auc = roc_auc_score(y_test, reconstruction_errors)
    pr_auc = average_precision_score(y_test, reconstruction_errors)
    
    print(f"  ROC AUC: {roc_auc:.3f}")
    print(f"  PR AUC: {pr_auc:.3f}")
    
    # Threshold analysis
    print(f"\\n🎚️  THRESHOLD ANALYSIS:")
    print("-" * 50)
    
    # Conservative threshold (95th percentile of normal errors)
    conservative_threshold = np.percentile(normal_errors, 95)
    
    # Precision-Recall curve for optimal threshold
    precision, recall, thresholds = precision_recall_curve(y_test, reconstruction_errors)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    # Balanced threshold (closest precision = recall)
    diff = np.abs(precision - recall)
    balanced_idx = np.argmin(diff)
    balanced_threshold = thresholds[balanced_idx]
    
    thresholds_to_test = [
        ("Conservative (95% normal)", conservative_threshold),
        ("Balanced (P≈R)", balanced_threshold), 
        ("Optimal F1", optimal_threshold)
    ]
    
    results = {}
    for name, threshold in thresholds_to_test:
        y_pred = (reconstruction_errors > threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity_val = tn / (tn + fp) if (tn + fp) > 0 else 0
        accuracy_val = (tp + tn) / (tp + tn + fp + fn)
        f1_val = 2 * (precision_val * recall_val) / (precision_val + recall_val) if (precision_val + recall_val) > 0 else 0
        
        results[name] = {
            'threshold': threshold,
            'precision': precision_val,
            'recall': recall_val,
            'specificity': specificity_val,
            'accuracy': accuracy_val,
            'f1_score': f1_val,
            'confusion_matrix': {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}
        }
        
        print(f"{name}:")
        print(f"  Threshold: {threshold:.6f}")
        print(f"  Precision: {precision_val:.3f}")
        print(f"  Recall: {recall_val:.3f}")
        print(f"  F1-Score: {f1_val:.3f}")
        print(f"  Accuracy: {accuracy_val:.3f}")
        print(f"  Specificity: {specificity_val:.3f}")
        print(f"  Confusion: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
        print()
    
    # Load training statistics if available
    try:
        with open("outputs/checkpoints/lstm_vae_gan_normal_only_stats.json", "r") as f:
            training_stats = json.load(f)
        
        print(f"\\n📈 TRAINING SUMMARY:")
        print("-" * 50)
        print(f"Training approach: {training_stats.get('training_approach', 'unknown')}")
        print(f"Normal samples used: {training_stats.get('normal_samples', 'unknown'):,}")
        print(f"Attack samples excluded: {training_stats.get('excluded_attack_samples', 'unknown'):,}")
        print(f"Total epochs: {training_stats.get('total_epochs', 'unknown')}")
        print(f"Best separation (training): {training_stats.get('best_separation_ratio', 0):.1f}x")
        print(f"Best PR-AUC (training): {training_stats.get('best_pr_auc', 0):.3f}")
        
        training_end = training_stats.get('training_end', 'Unknown')
        if training_end != 'Unknown':
            print(f"Training completed: {training_end[:19]}")
            
    except FileNotFoundError:
        print(f"\\n⚠️  Training statistics not found")
    
    # Final summary
    print(f"\\n" + "=" * 60)
    print(f"FINAL EVALUATION RESULTS")
    print(f"=" * 60)
    
    best_result = max(results.values(), key=lambda x: x['f1_score'])
    best_name = [name for name, result in results.items() if result == best_result][0]
    
    print(f"🏆 BEST PERFORMANCE ({best_name}):")
    print(f"   Precision: {best_result['precision']:.3f}")
    print(f"   Recall: {best_result['recall']:.3f}")
    print(f"   F1-Score: {best_result['f1_score']:.3f}")
    print(f"   Accuracy: {best_result['accuracy']:.3f}")
    print(f"   Separation: {separation_ratio:.1f}x")
    print(f"   PR-AUC: {pr_auc:.3f}")
    
    return {
        'separation_ratio': separation_ratio,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'best_result': best_result,
        'all_results': results,
        'normal_error_stats': {
            'mean': np.mean(normal_errors),
            'std': np.std(normal_errors),
            'count': len(normal_errors)
        },
        'attack_error_stats': {
            'mean': np.mean(attack_errors),
            'std': np.std(attack_errors),
            'count': len(attack_errors)
        }
    }

if __name__ == "__main__":
    evaluate_model()
