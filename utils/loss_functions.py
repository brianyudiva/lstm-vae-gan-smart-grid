"""
Loss functions for LSTM-VAE-GAN anomaly detection
"""
import tensorflow as tf
import numpy as np


def kl_loss(z_mean, z_log_var):
    """KL divergence loss for VAE"""
    return -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))


def reconstruction_loss(y_true, y_pred):
    """Standard MSE reconstruction loss"""
    return tf.reduce_mean(tf.square(y_true - y_pred))


def spectral_reconstruction_loss(y_true, y_pred):
    """Enhanced reconstruction loss with multiple components"""
    # Ensure both inputs are float32
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Standard MSE loss
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    
    # L1 loss for sparsity
    mae_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
    
    # Temporal consistency loss (penalize sudden changes)
    temporal_diff_true = y_true[:, 1:, :] - y_true[:, :-1, :]
    temporal_diff_pred = y_pred[:, 1:, :] - y_pred[:, :-1, :]
    temporal_loss = tf.reduce_mean(tf.square(temporal_diff_true - temporal_diff_pred))
    
    # Feature correlation loss (maintain relationships between features)
    feature_mean_true = tf.reduce_mean(y_true, axis=1, keepdims=True)
    feature_mean_pred = tf.reduce_mean(y_pred, axis=1, keepdims=True)
    feature_corr_true = feature_mean_true - y_true
    feature_corr_pred = feature_mean_pred - y_pred
    correlation_loss = tf.reduce_mean(tf.square(feature_corr_true - feature_corr_pred))
    
    return mse_loss + 0.1 * mae_loss + 0.10 * temporal_loss + 0.05 * correlation_loss


def robust_reconstruction_loss(y_true, y_pred, epsilon=0.01):
    """Simplified robust loss (Huber-style)"""
    diff = y_true - y_pred
    squared_diff = tf.square(diff)
    abs_diff = tf.abs(diff)
    
    # Huber loss
    huber_loss = tf.where(
        abs_diff <= epsilon,
        0.5 * squared_diff,
        epsilon * abs_diff - 0.5 * epsilon**2
    )
    
    return tf.reduce_mean(huber_loss)


def regularization_loss(encoder, decoder):
    """Enhanced regularization for better generalization"""
    # L2 regularization on weights
    l2_loss = 0
    for layer in encoder.layers:
        if hasattr(layer, 'kernel'):
            l2_loss += tf.reduce_sum(tf.square(layer.kernel))
        if hasattr(layer, 'bias') and layer.bias is not None:
            l2_loss += tf.reduce_sum(tf.square(layer.bias))
    
    for layer in decoder.layers:
        if hasattr(layer, 'kernel'):
            l2_loss += tf.reduce_sum(tf.square(layer.kernel))
        if hasattr(layer, 'bias') and layer.bias is not None:
            l2_loss += tf.reduce_sum(tf.square(layer.bias))
    
    # Stronger regularization for better anomaly detection
    return l2_loss * 0.01


def beta_schedule(epoch, total_epochs, max_beta=1.0, warmup_epochs=8):
    """Balanced beta scheduling for effective anomaly detection"""
    if epoch < warmup_epochs:
        # Moderate warmup to ensure reconstruction quality first
        return 0.0001 + (max_beta - 0.0001) * (epoch / warmup_epochs) ** 2  # Quadratic warmup
    else:
        # Stable beta after warmup
        return max_beta


def contrastive_latent_loss(z_normal, z_augmented, temperature=0.5):
    """Contrastive learning loss to cluster normal data tightly"""
    # Normalize latent vectors
    z_normal_norm = tf.nn.l2_normalize(z_normal, axis=1)
    z_augmented_norm = tf.nn.l2_normalize(z_augmented, axis=1)
    
    # Compute similarity matrix
    similarity = tf.matmul(z_normal_norm, z_augmented_norm, transpose_b=True) / temperature
    
    # Positive pairs should be similar (diagonal)
    batch_size = tf.shape(z_normal)[0]
    labels = tf.range(batch_size)
    
    # Cross-entropy loss for contrastive learning
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, similarity)
    return tf.reduce_mean(loss)


def focal_bce(y_true, y_pred, alpha=0.85, gamma=2.5):
    """
    Focal binary cross-entropy loss for handling class imbalance
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        alpha: Weighting factor for rare class (default 0.75)
        gamma: Focusing parameter (default 2.0)
        
    Returns:
        Focal loss value
    """
    # Ensure inputs are float32
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Clip predictions to prevent log(0)
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    
    # Calculate binary cross-entropy
    bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
    
    # Calculate p_t
    p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    
    # Calculate alpha_t (class balancing)
    alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
    
    # Calculate focal weight
    focal_weight = alpha_t * tf.pow((1 - p_t), gamma)
    
    # Apply focal weight to BCE
    focal_loss = focal_weight * bce
    
    return tf.reduce_mean(focal_loss)


def combined_discriminator_loss(y_true_stealth, y_pred_stealth, y_true_extreme, y_pred_extreme):
    """
    Combined discriminator loss applying focal BCE for stealth FDIA and regular BCE for extreme attacks
    
    Args:
        y_true_stealth: True labels for stealth FDIA samples
        y_pred_stealth: Predicted probabilities for stealth FDIA samples  
        y_true_extreme: True labels for extreme attack samples
        y_pred_extreme: Predicted probabilities for extreme attack samples
        
    Returns:
        Combined loss value
    """
    # Apply focal BCE for stealth FDIA samples (harder to detect)
    stealth_loss = focal_bce(y_true_stealth, y_pred_stealth, alpha=0.85, gamma=2.5)
    
    # Apply regular BCE for extreme attacks (easier to detect)
    extreme_loss = tf.keras.backend.binary_crossentropy(y_true_extreme, y_pred_extreme)
    extreme_loss = tf.reduce_mean(extreme_loss)
    
    # Combine losses with appropriate weighting
    # Give more weight to stealth detection since it's harder
    total_loss = 0.85 * stealth_loss + 0.15 * extreme_loss
    
    return total_loss


def precision_at_recall(y_true, y_scores, target_recall=0.6):
    """
    Calculate precision at a specific recall threshold
    
    Args:
        y_true: True binary labels
        y_scores: Predicted scores/probabilities
        target_recall: Target recall value (default 0.6)
        
    Returns:
        Precision value at target recall
    """
    # Convert to numpy if tensors
    if hasattr(y_true, 'numpy'):
        y_true = y_true.numpy()
    if hasattr(y_scores, 'numpy'):
        y_scores = y_scores.numpy()
    
    # Sort by scores in descending order
    sorted_indices = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[sorted_indices]
    
    # Calculate cumulative precision and recall
    true_positives = np.cumsum(y_true_sorted)
    total_positives = np.sum(y_true)
    
    if total_positives == 0:
        return 0.0
    
    recalls = true_positives / total_positives
    precisions = true_positives / np.arange(1, len(y_true_sorted) + 1)
    
    # Find the index where recall first exceeds target
    recall_indices = np.where(recalls >= target_recall)[0]
    
    if len(recall_indices) == 0:
        return 0.0
    
    # Return precision at the first point where recall >= target_recall
    return precisions[recall_indices[0]]


def anomaly_regularization_loss(z_mean, z_log_var, latent_samples):
    """Enhanced anomaly detection regularization with multiple strategies"""
    # Standard KL divergence to unit Gaussian
    kl_divergence = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    
    # Tight clustering of normal data around learned center
    latent_center = tf.reduce_mean(latent_samples, axis=0, keepdims=True)
    center_distances = tf.reduce_sum(tf.square(latent_samples - latent_center), axis=1)
    compactness_loss = tf.reduce_mean(center_distances)
    
    # Encourage small latent magnitude (stay near origin)
    magnitude_loss = tf.reduce_mean(tf.reduce_sum(tf.square(latent_samples), axis=1))
    
    # Minimize latent variance (tighter clustering)
    latent_var = tf.reduce_mean(tf.square(latent_samples - latent_center))
    
    # Hypersphere constraint - normal data should stay within a certain radius
    # Use a fixed radius instead of trainable variable to avoid variable scope issues
    target_radius = 1.5  # Fixed target radius
    sphere_loss = tf.reduce_mean(tf.maximum(0.0, center_distances - target_radius**2))
    
    return (kl_divergence + 
            1.5 * compactness_loss +      # Increased clustering
            0.5 * magnitude_loss +        # Keep near origin  
            0.8 * latent_var +           # Reduce variance
            0.3 * sphere_loss)           # Hypersphere constraint
