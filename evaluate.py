import json
import numpy as np
import os
import tensorflow as tf
from models.lstm_vae_gan import build_lstm_vae_gan_regular
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score
from sklearn.metrics import confusion_matrix, classification_report

def load_best_hyperparameters():
    """Load the best hyperparameters from configuration file."""
    try:
        with open("best_hyperparameters.json", "r") as f:
            config = json.load(f)
        return config.get("optimized_hyperparameters", {})
    except FileNotFoundError:
        return None

def evaluate_model():
    print("=" * 60)
    print("🔬 LSTM VAE GAN MODEL EVALUATION")
    print("=" * 60)

    sequence_path = "data/sequences"
    try:
        X_test = np.load(f"{sequence_path}/X_test.npy")
        y_test = np.load(f"{sequence_path}/y_test_binary.npy")
        print(f"✅ Loaded attack-preserved test data")
    except FileNotFoundError:
        print("❌ Error: Attack-preserved test data not found.")
        print("   Run preprocessing/attack_preserving_preprocessing.py first")
        return None
        
    model_files = [
        ("outputs/checkpoints/lstm_vae_gan_quick_optuna_encoder.h5", 
         "outputs/checkpoints/lstm_vae_gan_quick_optuna_decoder.h5", "Optuna-optimized"),
        ("outputs/checkpoints/lstm_vae_gan_encoder.h5", 
         "outputs/checkpoints/lstm_vae_gan_decoder.h5", "Standard")
    ]
    
    encoder_path = None
    decoder_path = None
    model_type = None
    
    for enc_path, dec_path, mtype in model_files:
        if os.path.exists(enc_path) and os.path.exists(dec_path):
            encoder_path = enc_path
            decoder_path = dec_path
            model_type = mtype
            break
    
    if encoder_path is None:
        print("❌ Error: No trained model found.")
        print("   Expected files:")
        for enc_path, dec_path, mtype in model_files:
            print(f"   - {mtype}: {enc_path}, {dec_path}")
        return None
    
    print(f"✅ Found {model_type} model files")
    
    # Show best hyperparameters if available and detect correct latent_dim
    best_config = load_best_hyperparameters()
    latent_dim = 32  # Default fallback
    
    # Try to get correct latent_dim from the specific model results
    if model_type == "Optuna-optimized":
        try:
            # Check Optuna results for actual latent_dim used
            optuna_results_file = "outputs/checkpoints/lstm_vae_gan_quick_optuna_results.json"
            with open(optuna_results_file, 'r') as f:
                optuna_results = json.load(f)
            if 'best_params' in optuna_results and 'latent_dim' in optuna_results['best_params']:
                latent_dim = optuna_results['best_params']['latent_dim']
                print(f"📋 Using Optuna model latent_dim: {latent_dim}")
            else:
                print(f"⚠️  Could not find latent_dim in Optuna results, using default: {latent_dim}")
        except FileNotFoundError:
            print(f"⚠️  Optuna results file not found, using default latent_dim: {latent_dim}")
    elif best_config and 'parameters' in best_config:
        latent_dim = best_config['parameters'].get('latent_dim', 16)
        print(f"📋 Using config latent_dim: {latent_dim}")
    
    if best_config and model_type == "Optuna-optimized":
        print(f"📊 Model info - PR-AUC: {best_config.get('best_pr_auc', 'unknown')}, latent_dim: {latent_dim}")
        if 'parameters' in best_config:
            params = best_config['parameters']
            print(f"   learning_rate: {params.get('learning_rate', 'unknown'):.6f}")
            print(f"   kl_weight: {params.get('kl_weight', 'unknown'):.6f}")
    
    input_shape = (X_test.shape[1], X_test.shape[2])
    encoder, decoder, _ = build_lstm_vae_gan_regular(
        input_shape=input_shape,
        latent_dim=latent_dim,  # Use detected latent_dim
    )
    
    encoder.load_weights(encoder_path)
    decoder.load_weights(decoder_path)
    print(f"✅ Loaded model weights")
    
    print(f"\n🔍 CALCULATING RECONSTRUCTION ERRORS...")
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
    
    normal_errors = reconstruction_errors[y_test == 0]
    attack_errors = reconstruction_errors[y_test == 1]
    
    print(f"\nRECONSTRUCTION ERROR ANALYSIS:")
    print("-" * 50)
    print(f"Normal samples:")
    print(f"  Count: {len(normal_errors)}")
    print(f"  Mean error: {np.mean(normal_errors):.6f}")
    print(f"  Std error: {np.std(normal_errors):.6f}")
    print(f"  Range: [{np.min(normal_errors):.6f}, {np.max(normal_errors):.6f}]")
    
    print(f"\nAttack samples:")
    print(f"  Count: {len(attack_errors)}")
    print(f"  Mean error: {np.mean(attack_errors):.6f}")
    print(f"  Std error: {np.std(attack_errors):.6f}")
    print(f"  Range: [{np.min(attack_errors):.6f}, {np.max(attack_errors):.6f}]")
    
    # Calculate separation ratio
    separation_ratio = np.mean(attack_errors) / np.mean(normal_errors)
    print(f"\nSEPARATION ANALYSIS:")
    print(f"  Separation ratio: {separation_ratio:.1f}x")
    
    # Performance metrics
    roc_auc = roc_auc_score(y_test, reconstruction_errors)
    pr_auc = average_precision_score(y_test, reconstruction_errors)
    
    print(f"  ROC AUC: {roc_auc:.3f}")
    print(f"  PR AUC: {pr_auc:.3f}")
    
    # Threshold analysis
    print(f"\nTHRESHOLD ANALYSIS:")
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
    stats_files = [
        ("outputs/checkpoints/lstm_vae_gan_quick_optuna_results.json", "Optuna optimization results"),
        ("outputs/checkpoints/lstm_vae_gan_stats.json", "Standard training stats"),
        ("outputs/checkpoints/lstm_vae_gan_normal_only_stats.json", "Legacy training stats")
    ]
    
    training_stats = None
    for stats_file, description in stats_files:
        try:
            with open(stats_file, "r") as f:
                training_stats = json.load(f)
            stats_type = description
            break
        except FileNotFoundError:
            continue
    
    if training_stats:
        print(f"\nTRAINING SUMMARY ({stats_type}):")
        print("-" * 50)
        
        # Handle different stats file formats
        if 'best_params' in training_stats:  # Optuna results
            print(f"Training approach: Optuna-optimized")
            print(f"Number of trials: {training_stats.get('n_trials', 'unknown')}")
            print(f"Best PR-AUC: {training_stats.get('best_value', training_stats.get('final_pr_auc', 'unknown')):.3f}")
            if 'best_params' in training_stats:
                print(f"Optimized parameters:")
                for param, value in training_stats['best_params'].items():
                    if isinstance(value, float):
                        print(f"   {param}: {value:.6f}")
                    else:
                        print(f"   {param}: {value}")
        else:  # Standard training stats
            print(f"Training approach: {training_stats.get('training_approach', 'standard')}")
            print(f"Normal samples used: {training_stats.get('normal_samples', 'unknown'):,}")
            print(f"Attack samples excluded: {training_stats.get('excluded_attack_samples', 'unknown'):,}")
            print(f"Total epochs: {training_stats.get('total_epochs', 'unknown')}")
            print(f"Best separation (training): {training_stats.get('best_separation_ratio', 0):.1f}x")
            print(f"Best PR-AUC (training): {training_stats.get('best_pr_auc', 0):.3f}")
            
            training_end = training_stats.get('training_end', 'Unknown')
            if training_end != 'Unknown':
                print(f"Training completed: {training_end[:19]}")
    else:
        print(f"\nNo training statistics found")
    
    # Final summary
    print(f"\n" + "=" * 60)
    print(f"FINAL EVALUATION RESULTS ({model_type} Model)")
    print(f"=" * 60)
    
    best_result = max(results.values(), key=lambda x: x['f1_score'])
    best_name = [name for name, result in results.items() if result == best_result][0]
    
    print(f"🏆 BEST PERFORMANCE ({best_name}):")
    print(f"   Precision: {best_result['precision']*100:.5f}%")
    print(f"   Recall: {best_result['recall']*100:.5f}%")
    print(f"   F1-Score: {best_result['f1_score']*100:.5f}%")
    print(f"   Accuracy: {best_result['accuracy']*100:.5f}%")
    print(f"   Separation: {separation_ratio:.5f}x")
    print(f"   PR-AUC: {pr_auc*100:.5f}%")
    
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
