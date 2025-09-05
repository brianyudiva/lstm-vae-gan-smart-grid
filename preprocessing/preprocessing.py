import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

def preprocess_with_zscore_all_data():
    # === CONFIG ===
    sequence_length = 12
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

    # === BUILD SEQUENCES FIRST (BEFORE SCALING) ===
    print(f"\n🔗 BUILDING SEQUENCES (before scaling):")
    X_raw, y_fdia = [], []
    total_sequences = len(all_features) - sequence_length + 1

    for i in range(total_sequences):
        X_seq = all_features[i:i + sequence_length]
        y_seq = int(labels[i + sequence_length - 1])  # Label based on last timestep
        
        X_raw.append(X_seq)
        y_fdia.append(y_seq)

    X_raw = np.array(X_raw)
    y_fdia = np.array(y_fdia)

    print(f"Created {len(X_raw)} sequences of length {sequence_length}")
    print(f"Attack sequences: {np.sum(y_fdia)}/{len(y_fdia)} ({np.sum(y_fdia)/len(y_fdia)*100:.1f}%)")

    # === TRAIN/TEST SPLIT (BEFORE SCALING) ===
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y_fdia, test_size=0.2, random_state=42, stratify=y_fdia
    )

    print(f"\nTRAIN/TEST SPLIT (before scaling):")
    print(f"Train: {len(X_train_raw)} sequences, {np.sum(y_train)} attacks ({np.sum(y_train)/len(y_train)*100:.1f}%)")
    print(f"Test:  {len(X_test_raw)} sequences, {np.sum(y_test)} attacks ({np.sum(y_test)/len(y_test)*100:.1f}%)")

    # === FIT SCALER ON TRAIN DATA ONLY ===
    print(f"\n📏 FITTING SCALER ON TRAIN DATA ONLY:")
    
    # Reshape training data to 2D for scaler fitting
    X_train_flat = X_train_raw.reshape(-1, X_train_raw.shape[-1])
    
    # Calculate pre-scaling statistics on training data
    normal_mask_train = y_train == 0
    attack_mask_train = y_train == 1
    
    X_train_normal_flat = X_train_raw[normal_mask_train].reshape(-1, X_train_raw.shape[-1])
    X_train_attack_flat = X_train_raw[attack_mask_train].reshape(-1, X_train_raw.shape[-1])
    
    # Only use z_attacked features for signal analysis
    n_z_features = len(z_attacked_cols)
    normal_features_train = X_train_normal_flat[:, :n_z_features]
    attack_features_train = X_train_attack_flat[:, :n_z_features]
    
    attack_signal_strength = np.abs(np.mean(attack_features_train) - np.mean(normal_features_train))
    print(f"Original attack signal strength (train only): {attack_signal_strength:.6f}")
    
    # Fit scaler on training data only
    scaler = StandardScaler()
    scaler.fit(X_train_flat)
    
    # Transform both train and test data
    X_train_flat_scaled = scaler.transform(X_train_flat)
    X_test_flat_scaled = scaler.transform(X_test_raw.reshape(-1, X_test_raw.shape[-1]))
    
    # Reshape back to sequences
    X_train = X_train_flat_scaled.reshape(X_train_raw.shape)
    X_test = X_test_flat_scaled.reshape(X_test_raw.shape)
    
    # === POST-SCALING ANALYSIS ===
    print(f"\nPOST-SCALING ANALYSIS:")
    
    # Analysis on scaled training data
    X_train_normal_scaled = X_train[y_train == 0]
    X_train_attack_scaled = X_train[y_train == 1]
    
    X_train_normal_flat_scaled = X_train_normal_scaled.reshape(-1, X_train.shape[-1])[:, :n_z_features]
    X_train_attack_flat_scaled = X_train_attack_scaled.reshape(-1, X_train.shape[-1])[:, :n_z_features]
    
    print(f"Train data - Mean: {np.mean(X_train_flat_scaled):.6f}, Std: {np.std(X_train_flat_scaled):.6f}")
    print(f"Train normal - Mean: {np.mean(X_train_normal_flat_scaled):.6f}, Std: {np.std(X_train_normal_flat_scaled):.6f}")
    print(f"Train attack - Mean: {np.mean(X_train_attack_flat_scaled):.6f}, Std: {np.std(X_train_attack_flat_scaled):.6f}")
    
    preserved_signal_strength = np.abs(np.mean(X_train_attack_flat_scaled) - np.mean(X_train_normal_flat_scaled))
    print(f"Preserved attack signal: {preserved_signal_strength:.6f}")
    
    if attack_signal_strength > 0:
        print(f"Signal preservation ratio: {preserved_signal_strength/attack_signal_strength:.3f}")
    
    normal_std = np.std(X_train_normal_flat_scaled)
    attack_deviation = preserved_signal_strength / normal_std if normal_std > 0 else 0
    print(f"Attack detectability: {attack_deviation:.3f} standard deviations")
    
    # Analysis on test data
    X_test_normal_scaled = X_test[y_test == 0]
    X_test_attack_scaled = X_test[y_test == 1]
    
    test_normal_mean = np.mean(X_test_normal_scaled)
    test_attack_mean = np.mean(X_test_attack_scaled)
    
    print(f"\nTEST DATA ANALYSIS:")
    print(f"Test data - Mean: {np.mean(X_test_flat_scaled):.6f}, Std: {np.std(X_test_flat_scaled):.6f}")
    print(f"Test normal sequences: {test_normal_mean:.6f}")
    print(f"Test attack sequences: {test_attack_mean:.6f}")
    
    test_attack_strength = np.abs(test_attack_mean - test_normal_mean)
    test_normal_std = np.std(X_test_normal_scaled)
    test_detectability = test_attack_strength / test_normal_std if test_normal_std > 0 else 0
    
    print(f"Test attack strength: {test_attack_strength:.6f}")
    print(f"Test detectability: {test_detectability:.3f} standard deviations")
    
    # Save scaler (fitted on train data only)
    joblib.dump(scaler, f"{sequence_dir}/scaler_zscore_train_only.pkl")
    
    print(f"\n💾 SAVING SEQUENCES:")
    print(f"Scaler fitted on: TRAIN DATA ONLY ({len(X_train_flat)} samples)")
    print(f"No data leakage from test set")

    # Save sequences
    np.save(f"{sequence_dir}/X_train.npy", X_train)
    np.save(f"{sequence_dir}/X_test.npy", X_test)
    np.save(f"{sequence_dir}/y_train_binary.npy", y_train)
    np.save(f"{sequence_dir}/y_test_binary.npy", y_test)

    # Create quality assessment
    quality_report = {
        'preprocessing_method': 'zscore_train_only_no_leakage',
        'scaler_fitted_on': 'train_data_only',
        'data_leakage_prevented': True,
        'original_attack_signal_train': float(attack_signal_strength),
        'preserved_attack_signal_train': float(preserved_signal_strength),
        'signal_preservation_ratio': float(preserved_signal_strength/attack_signal_strength) if attack_signal_strength > 0 else 0,
        'train_attack_detectability_stddevs': float(attack_deviation),
        'test_attack_detectability_stddevs': float(test_detectability),
        'train_sequences': len(X_train),
        'test_sequences': len(X_test),
        'train_attack_ratio': float(np.sum(y_train)/len(y_train)),
        'test_attack_ratio': float(np.sum(y_test)/len(y_test)),
        'sequence_labeling': 'last_timestep'
    }
    
    import json
    with open(f"{sequence_dir}/preprocessing_no_leakage_report.json", 'w') as f:
        json.dump(quality_report, f, indent=2)

    print(f"\n✅ PREPROCESSING COMPLETE - NO DATA LEAKAGE")
    print(f"Scaler file: scaler_zscore_train_only.pkl")
    print(f"Report: preprocessing_no_leakage_report.json")

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    preprocess_with_zscore_all_data()