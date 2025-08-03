import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import os

# === CONFIG ===
sequence_length = 12  # 12 hours of data
input_csv = "data/processed/ieee13_fdia.csv"
sequence_dir = "data/sequences"
os.makedirs(sequence_dir, exist_ok=True)

print("Loading FDIA dataset...")
df = pd.read_csv(input_csv)
print(f"Loaded {len(df)} records")

# Verify data integrity
required_cols = ["hour", "day", "fdia", "fdia_type"]
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

print(f"Data validation passed")

# Sort chronologically (day first, then hour) - CRITICAL FIX
print("Sorting data chronologically...")
df = df.sort_values(by=["day", "hour"]).reset_index(drop=True)

# Check for time gaps
expected_records = df["day"].max() * 24 + 24
if len(df) != expected_records:
    print(f"Warning: Expected {expected_records} records, got {len(df)}")

# Remove metadata columns for features
df_clean = df.drop(columns=["timestamp", "fdia_target_bus"])

# Extract labels before removing them from features
label_fdia = df_clean["fdia"].values
label_fdia_type = df_clean["fdia_type"].values

# Get feature columns (everything except labels and day/hour)
features = df_clean.drop(columns=["fdia", "fdia_type", "day", "hour"]).values

print(f"Feature shape: {features.shape}")
print(f"FDIA distribution: {np.sum(label_fdia)}/{len(label_fdia)} ({np.sum(label_fdia)/len(label_fdia)*100:.1f}%)")

# Scale features
print("Scaling features...")
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

if np.isnan(features).any():
    print("Warning: NaN values found in features! Check input data.")

# Save scaler for inference
import joblib
joblib.dump(scaler, f"{sequence_dir}/scaler.pkl")
print("Scaler saved for inference")

# === BUILD SEQUENCES ===
print(f"Building sequences of length {sequence_length}...")

X, y_fdia, y_fdia_type = [], [], []
total_sequences = len(features_scaled) - sequence_length + 1

for i in range(total_sequences):
    # Create sequence of features
    X_seq = features_scaled[i:i + sequence_length]
    
    # Label is from the LAST timestep in the sequence (anomaly detection)
    y_label_fdia = label_fdia[i + sequence_length - 1]
    y_label_type = label_fdia_type[i + sequence_length - 1]
    
    X.append(X_seq)
    y_fdia.append(y_label_fdia)
    y_fdia_type.append(y_label_type)
    
    # Progress indicator
    if (i + 1) % 500 == 0:
        progress = ((i + 1) / total_sequences) * 100
        print(f"   Progress: {progress:.1f}% ({i+1}/{total_sequences})")

X = np.array(X)
y_fdia = np.array(y_fdia)
y_fdia_type = np.array(y_fdia_type)

print(f"Created {len(X)} sequences")
print(f"Sequence FDIA distribution: {np.sum(y_fdia)}/{len(y_fdia)} ({np.sum(y_fdia)/len(y_fdia)*100:.1f}%)")

# Analyze sequence class balance
unique_types, type_counts = np.unique(y_fdia_type, return_counts=True)
print(f"FDIA Type distribution in sequences:")
for fdia_type, count in zip(unique_types, type_counts):
    type_name = ["Normal", "Voltage Spike", "Voltage Sag", "PV Manipulation", "Measurement Noise", "Coordinated Attack", "Stealthy Bias"][fdia_type]
    print(f"   Type {fdia_type} ({type_name}): {count} sequences ({count/len(y_fdia_type)*100:.1f}%)")

# Train/test split with stratification to preserve class balance
print("Splitting into train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_fdia, test_size=0.2, random_state=42, stratify=y_fdia
)

# Also split for multiclass labels (but don't need to stratify both)
X_train_type, X_test_type, y_train_type, y_test_type = train_test_split(
    X, y_fdia_type, test_size=0.2, random_state=42
)

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
print("\n" + "="*60)
print("SEQUENCE PREPROCESSING SUMMARY")
print("="*60)
print(f"Source data: {len(df)} hourly records")
print(f"Sequence length: {sequence_length} hours")
print(f"Total sequences: {len(X)}")
print(f"Input shape: {X.shape}")
print(f"Binary labels: {len(y_fdia)} (Normal: {np.sum(y_fdia == 0)}, FDIA: {np.sum(y_fdia == 1)})")
print(f"Multiclass labels: {len(y_fdia_type)} classes")
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
print(f"   FDIA training samples: {fdia_train_count} (will be excluded)")
print(f"   Effective training size: {normal_train_count}")

if normal_train_count < 100:
    print("WARNING: Very few normal samples for training!")
    
print(f"\nSequence preprocessing completed successfully!")