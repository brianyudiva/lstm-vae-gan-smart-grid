import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import seaborn as sns

# === LOAD MODELS ===
encoder = tf.keras.models.load_model("outputs/checkpoints/encoder.h5")
decoder = tf.keras.models.load_model("outputs/checkpoints/decoder.h5")

# === LOAD DATA ===
X_test = np.load("data/sequences/X_test.npy")
y_test = np.load("data/sequences/y_test_binary.npy")

# === RECONSTRUCTION ===
latent = encoder.predict(X_test)
X_reconstructed = decoder.predict(latent)

# === COMPUTE ERROR ===
recon_errors = np.mean((X_test - X_reconstructed) ** 2, axis=(1, 2))

# === THRESHOLDING ===
thresh = np.percentile(recon_errors, 95)  # top 5% errors as FDIA
preds = (recon_errors >= thresh).astype(int)

# === EVALUATION ===
print("ROC AUC:", roc_auc_score(y_test, recon_errors))
print(confusion_matrix(y_test, preds))
print(classification_report(y_test, preds))

# === VISUALIZATION ===
plt.figure(figsize=(10, 4))
sns.histplot(recon_errors[y_test == 0], label="Normal", color="blue", kde=True)
sns.histplot(recon_errors[y_test == 1], label="FDIA", color="red", kde=True)
plt.axvline(thresh, color="black", linestyle="--", label=f"Threshold = {thresh:.4f}")
plt.title("Reconstruction Error Distribution")
plt.xlabel("MSE")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.show()

# === REAL-TIME INFERENCE FUNCTION ===
def detect_fdia(input_sequence):
    input_sequence = np.expand_dims(input_sequence, axis=0)
    latent = encoder.predict(input_sequence)
    reconstructed = decoder.predict(latent)
    error = np.mean((input_sequence - reconstructed) ** 2)
    is_fdia = int(error >= thresh)
    return is_fdia, error

# === EXAMPLE USAGE ===
example = X_test[0]
flag, err = detect_fdia(example)
print(f"\nExample inference: Error = {err:.4f}, FDIA Detected = {bool(flag)}")
