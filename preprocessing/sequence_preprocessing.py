import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import os

# === CONFIG ===
sequence_length = 12
input_csv = "data/processed/ieee13_multitype_fdia.csv"
sequence_dir = "data/sequences"
os.makedirs(sequence_dir, exist_ok=True)

# === LOAD AND FILTER FEATURES ===
df = pd.read_csv(input_csv)

# Drop metadata not useful for input
df = df.drop(columns=["timestamp", "fdia_target_bus", "day"])

# Sort for safety (hour order)
df = df.sort_values(by=["hour"]).reset_index(drop=True)

# Extract labels and features
label_fdia = df["fdia"].values
label_fdia_type = df["fdia_type"].values
features = df.drop(columns=["fdia", "fdia_type"]).values

# Normalize features
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# === BUILD SEQUENCES ===
X, y_fdia, y_fdia_type = [], [], []
for i in range(len(features_scaled) - sequence_length):
    X_seq = features_scaled[i:i + sequence_length]
    y_label_fdia = label_fdia[i + sequence_length - 1]  # Binary FDIA
    y_label_type = label_fdia_type[i + sequence_length - 1]  # Multi-class
    X.append(X_seq)
    y_fdia.append(y_label_fdia)
    y_fdia_type.append(y_label_type)

X = np.array(X)
y_fdia = np.array(y_fdia)
y_fdia_type = np.array(y_fdia_type)

# === TRAIN-TEST SPLIT ===
X_train, X_test, y_train, y_test = train_test_split(X, y_fdia, test_size=0.2, random_state=42)
X_train_type, X_test_type, y_train_type, y_test_type = train_test_split(X, y_fdia_type, test_size=0.2, random_state=42)

# === SAVE ===
np.save(f"{sequence_dir}/X_fdia.npy", X)
np.save(f"{sequence_dir}/y_fdia_binary.npy", y_fdia)
np.save(f"{sequence_dir}/y_fdia_type.npy", y_fdia_type)

np.save(f"{sequence_dir}/X_train.npy", X_train)
np.save(f"{sequence_dir}/X_test.npy", X_test)
np.save(f"{sequence_dir}/y_train_binary.npy", y_train)
np.save(f"{sequence_dir}/y_test_binary.npy", y_test)
np.save(f"{sequence_dir}/y_train_type.npy", y_train_type)
np.save(f"{sequence_dir}/y_test_type.npy", y_test_type)

# === TF.DATA PIPELINE ===
def create_tf_dataset(X, y, batch_size=32, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X))
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

train_ds = create_tf_dataset(X_train, y_train)
test_ds = create_tf_dataset(X_test, y_test, shuffle=False)

# === PREVIEW ===
print("Data prepared and saved.")
print("X shape:", X.shape)
print("Train set:", X_train.shape, y_train.shape)
print("Test set:", X_test.shape, y_test.shape)
print("TF dataset example:", next(iter(train_ds))[0].shape)