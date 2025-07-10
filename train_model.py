import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping
import os

# === LOAD DATA ===
data_path = "data/sequences"
X_train = np.load(f"{data_path}/X_train.npy")
y_train = np.load(f"{data_path}/y_train_binary.npy")
X_test = np.load(f"{data_path}/X_test.npy")
y_test = np.load(f"{data_path}/y_test_binary.npy")

# === DEFINE MODEL ===
model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
    Dropout(0.3),
    LSTM(32),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

model.summary()

# === TRAIN ===
callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
]

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=callbacks
)

# === EVALUATE ===
loss, acc = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {acc:.4f}")

# === SAVE MODEL ===
os.makedirs("outputs/checkpoints", exist_ok=True)
model.save("outputs/checkpoints/lstm_fdia_model.h5")
