import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
import json

def preprocess():
    # === CONFIG ===
    sequence_length = 24
    input_csv = "data/processed/ieee13_fdia_dataset.csv"
    sequence_dir = "data/sequences"
    os.makedirs(sequence_dir, exist_ok=True)

    print("=" * 60)
    print("NORMALIZATION")
    print("=" * 60)

    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} records")

    required_cols = ["hour_of_day", "day", "fdia_label"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    print(f"FDIA distribution: {np.sum(df['fdia_label'])}/{len(df)} ({np.sum(df['fdia_label'])/len(df)*100:.1f}%)")

    df = df.sort_values(by=["day", "hour_of_day"]).reset_index(drop=True)

    z_attacked_cols = [col for col in df.columns if col.startswith('z_attacked_')]
    print(f"Found {len(z_attacked_cols)} measurement features")

    features_raw = df[z_attacked_cols].values
    labels = df["fdia_label"].values

    # Create time features
    hour_sin = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    hour_cos = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    day_sin = np.sin(2 * np.pi * df["day"] / 7)
    day_cos = np.cos(2 * np.pi * df["day"] / 7)
    time_features = np.column_stack([hour_sin, hour_cos, day_sin, day_cos])

    all_features = np.column_stack([features_raw, time_features])
    
    print(f"\nBUILDING SEQUENCES:")
    X_raw, y_fdia = [], []
    total_sequences = len(all_features) - sequence_length + 1

    for i in range(total_sequences):
        X_seq = all_features[i:i + sequence_length]
        y_seq = int(np.any(labels[i:i + sequence_length] == 1))

        X_raw.append(X_seq)
        y_fdia.append(y_seq)

    X_raw = np.array(X_raw)
    y_fdia = np.array(y_fdia)

    print(f"Created {len(X_raw)} sequences of length {sequence_length}")
    print(f"Attack sequences: {np.sum(y_fdia)}/{len(y_fdia)} ({np.sum(y_fdia)/len(y_fdia)*100:.1f}%)")

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y_fdia, test_size=0.2, random_state=42, stratify=y_fdia
    )

    print(f"\nTRAIN/TEST SPLIT (before scaling):")
    print(f"Train: {len(X_train_raw)} sequences")
    print(f"Test:  {len(X_test_raw)} sequences")

    X_train_flat = X_train_raw.reshape(-1, X_train_raw.shape[-1])

    scaler = StandardScaler()
    scaler.fit(X_train_flat)
    
    X_train_flat_scaled = scaler.transform(X_train_flat)
    X_test_flat_scaled = scaler.transform(X_test_raw.reshape(-1, X_test_raw.shape[-1]))
    
    X_train = X_train_flat_scaled.reshape(X_train_raw.shape)
    X_test = X_test_flat_scaled.reshape(X_test_raw.shape)    
    
    joblib.dump(scaler, f"{sequence_dir}/scaler.pkl")
    
    print(f"\nSAVING SEQUENCES:")
    print(f"Scaler fitted on: TRAIN DATA ONLY ({len(X_train_flat)} samples)")

    # Save sequences
    np.save(f"{sequence_dir}/X_train.npy", X_train)
    np.save(f"{sequence_dir}/X_test.npy", X_test)
    np.save(f"{sequence_dir}/y_train_binary.npy", y_train)
    np.save(f"{sequence_dir}/y_test_binary.npy", y_test)

    print(f"\nPREPROCESSING COMPLETE")

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    preprocess()