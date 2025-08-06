import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import os

# === CONFIG ===
sequence_length = 12  # 12 hours of data
input_csv = "data/processed/ieee13_jacobian_fdia_dataset.csv"
sequence_dir = "data/sequences"
os.makedirs(sequence_dir, exist_ok=True)

print("Loading Jacobian-based FDIA dataset...")
df = pd.read_csv(input_csv)
print(f"Loaded {len(df)} records")

# Verify data integrity for new format
required_cols = ["hour_of_day", "day", "fdia_label", "attack_magnitude"]
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

print(f"Data validation passed")
print(f"Dataset columns: {len(df.columns)}")
print(f"FDIA distribution: {np.sum(df['fdia_label'])}/{len(df)} ({np.sum(df['fdia_label'])/len(df)*100:.1f}%)")

# Sort chronologically (day first, then hour) - CRITICAL FIX
print("Sorting data chronologically...")
df = df.sort_values(by=["day", "hour_of_day"]).reset_index(drop=True)

# Check for time gaps
expected_records = df["day"].max() * 24 + 24  # Assuming hourly data
if len(df) != expected_records:
    print(f"Warning: Expected {expected_records} records, got {len(df)}")

# === FEATURE SELECTION STRATEGY ===
# For the Jacobian dataset, we have several options:
# 1. Use only z_attacked (simplest - what the system actually sees)
# 2. Use z_normal + attack_vector (for research analysis)  
# 3. Use all measurements (most complete but high-dimensional)

print("Selecting features for training...")

# Strategy 1: Use attacked measurements (what the system actually observes)
# This is most realistic - the model should detect attacks from corrupted measurements
z_attacked_cols = [col for col in df.columns if col.startswith('z_attacked_')]
print(f"Found {len(z_attacked_cols)} attacked measurement features")

# Extract metadata and labels
metadata_cols = ["timestep", "hour_of_day", "day", "load_factor", "attack_magnitude", "attack_stealth", "measurement_noise_std"]
label_col = "fdia_label"

# Create feature matrix from attacked measurements
features_raw = df[z_attacked_cols].values
labels = df[label_col].values

print(f"Raw feature shape: {features_raw.shape}")
print(f"Feature columns: z_attacked_0 to z_attacked_{len(z_attacked_cols)-1}")

# Add time-based periodic features (enhanced from previous version)
print("Adding time-based periodic features...")
hour_sin = np.sin(2 * np.pi * df["hour_of_day"] / 24)
hour_cos = np.cos(2 * np.pi * df["hour_of_day"] / 24)
day_sin = np.sin(2 * np.pi * df["day"] / 7)  # Weekly periodicity
day_cos = np.cos(2 * np.pi * df["day"] / 7)

# Also add load factor as it affects power system behavior
load_factor = df["load_factor"].values

# Stack all features together
time_features = np.column_stack([hour_sin, hour_cos, day_sin, day_cos, load_factor])
features = np.column_stack([features_raw, time_features])

print(f"Final feature shape: {features.shape}")
print(f"Features: {len(z_attacked_cols)} measurements + 5 engineered features")
print(f"   - {len(z_attacked_cols)} attacked measurements (z_attacked_*)")
print(f"   - 4 time-based periodic features (hour_sin/cos, day_sin/cos)")
print(f"   - 1 load factor feature")

# Check for data quality issues
if np.isnan(features).any():
    print("Warning: NaN values found in features! Checking...")
    nan_count = np.isnan(features).sum()
    print(f"Total NaN values: {nan_count}")
    
    # Fill NaN with column means as fallback
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    features = imputer.fit_transform(features)
    print("NaN values imputed with column means")

# Scale features
print("Scaling features...")
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# Save scaler for inference
joblib.dump(scaler, f"{sequence_dir}/scaler_jacobian.pkl")
print("Scaler saved for inference")

# === BUILD SEQUENCES ===
print(f"Building sequences of length {sequence_length}...")

X, y_fdia = [], []
total_sequences = len(features_scaled) - sequence_length + 1

for i in range(total_sequences):
    # Create sequence of features
    X_seq = features_scaled[i:i + sequence_length]
    
    # Label is from the LAST timestep in the sequence (anomaly detection)
    y_label_fdia = labels[i + sequence_length - 1]
    
    X.append(X_seq)
    y_fdia.append(y_label_fdia)
    
    # Progress indicator
    if (i + 1) % 1000 == 0:
        progress = ((i + 1) / total_sequences) * 100
        print(f"   Progress: {progress:.1f}% ({i+1}/{total_sequences})")

X = np.array(X)
y_fdia = np.array(y_fdia)

print(f"Created {len(X)} sequences")
print(f"Sequence FDIA distribution: {np.sum(y_fdia)}/{len(y_fdia)} ({np.sum(y_fdia)/len(y_fdia)*100:.1f}%)")

# For Jacobian attacks, we only have binary labels (no attack types like the old dataset)
# Create a placeholder for compatibility
y_fdia_type = y_fdia.copy()  # Same as binary for this dataset

print(f"Attack distribution in sequences:")
print(f"   Normal: {np.sum(y_fdia == 0)} sequences ({np.sum(y_fdia == 0)/len(y_fdia)*100:.1f}%)")
print(f"   FDIA: {np.sum(y_fdia == 1)} sequences ({np.sum(y_fdia == 1)/len(y_fdia)*100:.1f}%)")

# Train/test split with stratification to preserve class balance
print("Splitting into train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_fdia, test_size=0.2, random_state=42, stratify=y_fdia
)

# For compatibility with existing training scripts
y_train_type = y_train.copy()
y_test_type = y_test.copy()

print(f"Train set: {X_train.shape[0]} sequences ({np.sum(y_train)} FDIA)")
print(f"Test set: {X_test.shape[0]} sequences ({np.sum(y_test)} FDIA)")
print(f"Sequence shape: {X_train.shape[1:]}")

# === SAVE SEQUENCES ===
print("Saving sequence data...")

# Save complete dataset
np.save(f"{sequence_dir}/X_fdia.npy", X)
np.save(f"{sequence_dir}/y_fdia_binary.npy", y_fdia)
np.save(f"{sequence_dir}/y_fdia_type.npy", y_fdia_type)

# Save train/test splits
np.save(f"{sequence_dir}/X_train.npy", X_train)
np.save(f"{sequence_dir}/X_test.npy", X_test)
np.save(f"{sequence_dir}/y_train_binary.npy", y_train)
np.save(f"{sequence_dir}/y_test_binary.npy", y_test)
np.save(f"{sequence_dir}/y_train_type.npy", y_train_type)
np.save(f"{sequence_dir}/y_test_type.npy", y_test_type)

print("All sequences saved successfully!")

# === SUMMARY STATISTICS ===
print("\n" + "="*70)
print("JACOBIAN FDIA SEQUENCE PREPROCESSING SUMMARY")
print("="*70)
print(f"Source data: {len(df)} hourly records")
print(f"Sequence length: {sequence_length} hours")
print(f"Total sequences: {len(X)}")
print(f"Input shape: {X.shape}")
print(f"Feature composition:")
print(f"   - {len(z_attacked_cols)} attacked measurements (realistic)")
print(f"   - 5 engineered features (time + load)")
print(f"   - Total: {X.shape[2]} features per timestep")

print(f"\nLabels: {len(y_fdia)} (Normal: {np.sum(y_fdia == 0)}, FDIA: {np.sum(y_fdia == 1)})")
print(f"Attack type: Jacobian-based stealth attacks (a = Hc)")
print(f"Saved to: {sequence_dir}/")

print(f"\nTRAINING DATA:")
print(f"   Train sequences: {len(X_train)} ({np.sum(y_train)} FDIA)")
print(f"   Test sequences: {len(X_test)} ({np.sum(y_test)} FDIA)")
print(f"   Train FDIA rate: {np.sum(y_train)/len(y_train)*100:.1f}%")
print(f"   Test FDIA rate: {np.sum(y_test)/len(y_test)*100:.1f}%")

# Validate normal data for anomaly detection training
normal_train_count = np.sum(y_train == 0)
fdia_train_count = np.sum(y_train == 1)
print(f"\nANOMALY DETECTION SETUP:")
print(f"   Normal training samples: {normal_train_count}")
print(f"   FDIA samples available: {fdia_train_count}")
print(f"   Semi-supervised potential: {fdia_train_count} hard negatives")

if normal_train_count < 100:
    print("WARNING: Very few normal samples for training!")
    
print(f"\nSequence preprocessing completed successfully!")
