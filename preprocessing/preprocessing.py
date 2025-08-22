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
    print("Z-SCORE NORMALIZATION (ALL DATA)")
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

    hour_sin = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    hour_cos = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    day_sin = np.sin(2 * np.pi * df["day"] / 7)
    day_cos = np.cos(2 * np.pi * df["day"] / 7)
    time_features = np.column_stack([hour_sin, hour_cos, day_sin, day_cos])

    all_features = np.column_stack([features_raw, time_features])

    normal_mask = labels == 0
    attack_mask = labels == 1

    normal_features = features_raw[normal_mask]
    attack_features = features_raw[attack_mask]

    print(f"\nDATA OVERVIEW:")
    print(f"Normal samples: {len(normal_features)}")
    print(f"Attack samples: {len(attack_features)}")

    print(f"\nPRE-SCALING ANALYSIS:")
    print(f"Normal data - Mean: {np.mean(normal_features):.6f}, Std: {np.std(normal_features):.6f}")
    print(f"Attack data - Mean: {np.mean(attack_features):.6f}, Std: {np.std(attack_features):.6f}")
    attack_signal_strength = np.abs(np.mean(attack_features) - np.mean(normal_features))
    print(f"Attack signal strength: {attack_signal_strength:.6f}")
    
    scaler = StandardScaler()
    scaler.fit(all_features)
    
    features_scaled = scaler.transform(all_features)
    
    normal_scaled = features_scaled[normal_mask, :len(z_attacked_cols)]
    attack_scaled = features_scaled[attack_mask, :len(z_attacked_cols)]
    
    print(f"\nPOST-SCALING ANALYSIS:")
    print(f"All data - Mean: {np.mean(features_scaled):.6f}, Std: {np.std(features_scaled):.6f}")
    print(f"Normal scaled - Mean: {np.mean(normal_scaled):.6f}, Std: {np.std(normal_scaled):.6f}")
    print(f"Attack scaled - Mean: {np.mean(attack_scaled):.6f}, Std: {np.std(attack_scaled):.6f}")
    
    preserved_signal_strength = np.abs(np.mean(attack_scaled) - np.mean(normal_scaled))
    print(f"Preserved attack signal: {preserved_signal_strength:.6f}")
    
    if attack_signal_strength > 0:
        print(f"Signal preservation ratio: {preserved_signal_strength/attack_signal_strength:.3f}")
    
    normal_std = np.std(normal_scaled)
    attack_deviation = preserved_signal_strength / normal_std if normal_std > 0 else 0
    print(f"Attack detectability: {attack_deviation:.3f} standard deviations")
    
    # Save scaler
    joblib.dump(scaler, f"{sequence_dir}/scaler_zscore_all.pkl")
    
    # === BUILD SEQUENCES ===
    print(f"\n🔗 BUILDING SEQUENCES:")
    X, y_fdia = [], []
    total_sequences = len(features_scaled) - sequence_length + 1

    for i in range(total_sequences):
        X_seq = features_scaled[i:i + sequence_length]
        
        # Label based on ANY attack in the sequence
        y_seq = int(np.any(labels[i:i + sequence_length] == 1))
        
        X.append(X_seq)
        y_fdia.append(y_seq)

    X = np.array(X)
    y_fdia = np.array(y_fdia)

    print(f"Created {len(X)} sequences of length {sequence_length}")
    print(f"Attack sequences: {np.sum(y_fdia)}/{len(y_fdia)} ({np.sum(y_fdia)/len(y_fdia)*100:.1f}%)")

    # Split into train/test with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_fdia, test_size=0.2, random_state=42, stratify=y_fdia
    )

    print(f"\nTRAIN/TEST SPLIT:")
    print(f"Train: {len(X_train)} sequences, {np.sum(y_train)} attacks ({np.sum(y_train)/len(y_train)*100:.1f}%)")
    print(f"Test:  {len(X_test)} sequences, {np.sum(y_test)} attacks ({np.sum(y_test)/len(y_test)*100:.1f}%)")

    # Final validation of attack detectability in sequences
    X_train_normal = X_train[y_train == 0]
    X_test_attack = X_test[y_test == 1]
    X_test_normal = X_test[y_test == 0]
    
    print(f"\nSEQUENCE-LEVEL ATTACK ANALYSIS:")
    train_normal_mean = np.mean(X_train_normal)
    test_attack_mean = np.mean(X_test_attack)
    test_normal_mean = np.mean(X_test_normal)
    
    print(f"Train normal sequences: {train_normal_mean:.6f}")
    print(f"Test normal sequences:  {test_normal_mean:.6f}")
    print(f"Test attack sequences:  {test_attack_mean:.6f}")
    
    sequence_attack_strength = np.abs(test_attack_mean - test_normal_mean)
    sequence_normal_std = np.std(X_test_normal)
    sequence_detectability = sequence_attack_strength / sequence_normal_std if sequence_normal_std > 0 else 0
    
    print(f"Sequence attack strength: {sequence_attack_strength:.6f}")
    print(f"Sequence detectability: {sequence_detectability:.3f} standard deviations")

    # Save sequences with zscore suffix
    np.save(f"{sequence_dir}/X_train.npy", X_train)
    np.save(f"{sequence_dir}/X_test.npy", X_test)
    np.save(f"{sequence_dir}/y_train_binary.npy", y_train)
    np.save(f"{sequence_dir}/y_test_binary.npy", y_test)

    # Create quality assessment
    quality_report = {
        'preprocessing_method': 'zscore_all_data',
        'scaler_fitted_on': 'all_data_normal_and_attack',
        'original_attack_signal': float(attack_signal_strength),
        'preserved_attack_signal': float(preserved_signal_strength),
        'signal_preservation_ratio': float(preserved_signal_strength/attack_signal_strength) if attack_signal_strength > 0 else 0,
        'attack_detectability_stddevs': float(attack_deviation),
        'sequence_attack_detectability': float(sequence_detectability),
        'train_sequences': len(X_train),
        'test_sequences': len(X_test),
        'train_attack_ratio': float(np.sum(y_train)/len(y_train)),
        'test_attack_ratio': float(np.sum(y_test)/len(y_test))
    }
    
    import json
    with open(f"{sequence_dir}/preprocessing_zscore_report.json", 'w') as f:
        json.dump(quality_report, f, indent=2)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    preprocess_with_zscore_all_data()
