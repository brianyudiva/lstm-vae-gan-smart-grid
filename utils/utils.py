import numpy as np
import tensorflow as tf

def convert_to_json_serializable(obj):
    """
    Convert TensorFlow/numpy objects to JSON serializable Python types
    """
    if isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'numpy'):  # TensorFlow tensor
        return float(obj.numpy())
    elif isinstance(obj, (tf.Tensor, tf.Variable)):
        return float(obj.numpy())
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    else:
        return obj


def create_anomaly_labels(data_normal, data_anomaly):
    """
    Create binary labels for anomaly detection
    
    Args:
        data_normal: Normal/clean data samples
        data_anomaly: Anomaly/attack data samples
        
    Returns:
        Combined labels array (0 for normal, 1 for anomaly)
    """
    normal_labels = np.zeros(len(data_normal))
    anomaly_labels = np.ones(len(data_anomaly))
    return np.concatenate([normal_labels, anomaly_labels])