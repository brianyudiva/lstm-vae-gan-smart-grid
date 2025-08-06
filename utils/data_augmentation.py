"""
Data augmentation utilities for LSTM-VAE-GAN anomaly detection
"""
import numpy as np


def enhanced_data_augmentation(X_normal, augmentation_factor=0.3):
    """Enhanced data augmentation with adversarial-style perturbations
    
    Args:
        X_normal: Normal training data of shape (n_samples, seq_len, n_features)
        augmentation_factor: Fraction of original data to generate as augmentations
        
    Returns:
        Augmented data array
    """
    n_augmented = int(len(X_normal) * augmentation_factor)
    augmented_data = []
    
    # Get dynamic feature dimensions
    n_timesteps, n_features = X_normal.shape[1], X_normal.shape[2]
    
    for i in range(n_augmented):
        idx = np.random.randint(0, len(X_normal))
        sample = X_normal[idx].copy()
        
        # Mix of augmentation types
        aug_types = np.random.choice(['noise', 'temporal_shift', 'feature_dropout', 
                                    'temporal_jitter', 'amplitude_scale', 'adversarial_noise'], 
                                   size=np.random.randint(1, 3), replace=False)
        
        for aug_type in aug_types:
            if aug_type == 'noise':
                # Gaussian noise
                noise_level = np.random.uniform(0.005, 0.02)
                sample += np.random.normal(0, noise_level, sample.shape)
                
            elif aug_type == 'temporal_shift':
                # Time shifts
                shift = np.random.randint(-3, 4)
                if shift != 0:
                    sample = np.roll(sample, shift, axis=0)
                    
            elif aug_type == 'feature_dropout':
                # Feature masking (avoid time features which are last 4)
                n_mask_features = np.random.randint(1, min(4, n_features - 4))
                features_to_mask = np.random.choice(n_features - 4, size=n_mask_features, replace=False)
                time_steps = np.random.choice(n_timesteps, size=np.random.randint(2, min(5, n_timesteps)), replace=False)
                
                for feat in features_to_mask:
                    for t in time_steps:
                        if t > 0:
                            sample[t, feat] = sample[t-1, feat] * np.random.uniform(0.8, 1.2)
                            
            elif aug_type == 'temporal_jitter':
                # Temporal variations (avoid time features)
                for feat in range(n_features - 4):
                    if np.random.random() < 0.4:
                        jitter = np.random.normal(0, 0.003, n_timesteps)
                        sample[:, feat] += jitter
                        
            elif aug_type == 'amplitude_scale':
                # Scale certain features (avoid time features)
                features_to_scale = np.random.choice(n_features - 4, size=np.random.randint(2, min(5, n_features - 4)), replace=False)
                for feat in features_to_scale:
                    scale = np.random.uniform(0.9, 1.1)
                    sample[:, feat] *= scale
                    
            elif aug_type == 'adversarial_noise':
                # Small adversarial-like perturbations
                perturbation = np.random.uniform(-0.01, 0.01, sample.shape)
                sample += perturbation
        
        # Ensure we don't go too far from original distribution
        sample = np.clip(sample, X_normal.min() - 0.1, X_normal.max() + 0.1)
        augmented_data.append(sample)
    
    return np.array(augmented_data)


def create_synthetic_anomalies(X_normal, anomaly_factor=0.1):
    """Create synthetic anomalies for separation training (deprecated for pure anomaly detection)
    
    Args:
        X_normal: Normal training data
        anomaly_factor: Fraction of synthetic anomalies to create
        
    Returns:
        Synthetic anomaly data
    """
    n_anomalies = int(len(X_normal) * anomaly_factor)
    synthetic_anomalies = []
    
    # Get the actual feature dimensions from the input data
    n_timesteps, n_features = X_normal.shape[1], X_normal.shape[2]
    
    for i in range(n_anomalies):
        idx = np.random.randint(0, len(X_normal))
        sample = X_normal[idx].copy()
        
        # Create more extreme anomalies
        anomaly_type = np.random.choice(['extreme_noise', 'feature_corruption', 
                                       'temporal_anomaly', 'pattern_injection'])
        
        if anomaly_type == 'extreme_noise':
            # Add significant noise
            noise_level = np.random.uniform(0.05, 0.15)
            sample += np.random.normal(0, noise_level, sample.shape)
            
        elif anomaly_type == 'feature_corruption':
            # Corrupt entire features
            n_corrupt = np.random.randint(2, min(6, n_features))
            features_to_corrupt = np.random.choice(n_features, size=n_corrupt, replace=False)
            for feat in features_to_corrupt:
                corruption_type = np.random.choice(['zero', 'constant', 'spike'])
                if corruption_type == 'zero':
                    sample[:, feat] = 0
                elif corruption_type == 'constant':
                    sample[:, feat] = np.random.uniform(-1, 1)
                elif corruption_type == 'spike':
                    spike_positions = np.random.choice(n_timesteps, size=np.random.randint(2, min(5, n_timesteps)), replace=False)
                    for pos in spike_positions:
                        sample[pos, feat] *= np.random.uniform(3, 7)
                        
        elif anomaly_type == 'temporal_anomaly':
            # Create temporal anomalies
            anomaly_window = np.random.randint(3, min(8, n_timesteps))
            start_pos = np.random.randint(0, n_timesteps - anomaly_window)
            
            # Inject anomalous pattern
            for t in range(start_pos, start_pos + anomaly_window):
                sample[t, :] += np.random.normal(0, 0.1, n_features)
                sample[t, :] *= np.random.uniform(0.5, 2.0, n_features)
                
        elif anomaly_type == 'pattern_injection':
            # Inject completely different patterns
            pattern_length = np.random.randint(4, min(8, n_timesteps))
            start_pos = np.random.randint(0, n_timesteps - pattern_length)
            
            # Create sinusoidal or step patterns
            if np.random.random() > 0.5:
                # Sinusoidal
                t = np.linspace(0, 4*np.pi, pattern_length)
                for feat in range(n_features):
                    if np.random.random() > 0.5:
                        freq = np.random.uniform(0.5, 3.0)
                        amplitude = np.random.uniform(0.1, 0.5)
                        sample[start_pos:start_pos+pattern_length, feat] = amplitude * np.sin(freq * t)
            else:
                # Step pattern
                for feat in range(n_features):
                    if np.random.random() > 0.5:
                        step_value = np.random.uniform(-0.3, 0.3)
                        sample[start_pos:start_pos+pattern_length, feat] = step_value
        
        synthetic_anomalies.append(sample)
    
    return np.array(synthetic_anomalies)
