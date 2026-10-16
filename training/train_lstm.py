from pathlib import Path
import json
import numpy as np
import joblib
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from training.splits import split_data
from training.features import normalize_sequence, augment_sequence

ROOT = Path(__file__).resolve().parents[1]
data = np.load(ROOT / "data/processed/landmarks.npz", allow_pickle=True)
X, y = data["X"].astype("float32"), data["y"]
X = np.stack([normalize_sequence(s) for s in X])  # body-relative features: (N, 32, 351)
groups = data["signer"] if "signer" in data else None
tr, te, split_type = split_data(X, y, groups)

enc = LabelEncoder()
yt = enc.fit_transform(y)
num_classes = len(enc.classes_)

# ---------------------------------------------------------------------------
# Data Augmentation on Training Signers (5x data multiplier)
# ---------------------------------------------------------------------------
print(f"Original training samples: {len(tr)} across {num_classes} classes")
augmented_X = []
augmented_y = []

for idx in tr:
    orig = X[idx]
    label = yt[idx]
    # 1. Original
    augmented_X.append(orig)
    augmented_y.append(label)
    # 2. Scale + Jitter
    augmented_X.append(augment_sequence(orig, jitter=True, scale=True, rotate=False, speed=False, drop_frame=False))
    augmented_y.append(label)
    # 3. Rotate + Jitter
    augmented_X.append(augment_sequence(orig, jitter=True, scale=False, rotate=True, speed=False, drop_frame=False))
    augmented_y.append(label)
    # 4. Speed Resampling
    augmented_X.append(augment_sequence(orig, jitter=False, scale=False, rotate=False, speed=True, drop_frame=False))
    augmented_y.append(label)
    # 5. Combined Frame Dropout + subtle Scale/Jitter
    augmented_X.append(augment_sequence(orig, jitter=True, scale=True, rotate=True, speed=False, drop_frame=True))
    augmented_y.append(label)

X_train_aug = np.stack(augmented_X).astype(np.float32)
y_train_aug = np.array(augmented_y, dtype=np.int32)
print(f"Augmented training samples: {len(X_train_aug)}")

# Shuffle augmented training set
rng = np.random.default_rng(42)
perm = rng.permutation(len(X_train_aug))
X_train_aug = X_train_aug[perm]
y_train_aug = y_train_aug[perm]

# Split 15% for validation monitoring
n_val = max(1, int(0.15 * len(X_train_aug)))
val_idx, fit_idx = np.arange(n_val), np.arange(n_val, len(X_train_aug))

# Convert to categorical targets with label smoothing
y_train_cat = tf.keras.utils.to_categorical(y_train_aug, num_classes=num_classes)

weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train_aug[fit_idx]), y=y_train_aug[fit_idx])
class_weight = {int(c): float(w) for c, w in zip(np.unique(y_train_aug[fit_idx]), weights)}

# ---------------------------------------------------------------------------
# Conv1D + Bidirectional LSTM + Attention/Dual-Pooling Architecture
# ---------------------------------------------------------------------------
def build_model(input_shape, num_classes):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Masking(mask_value=0.0)(inputs)
    
    # 1. 1D Convolutions for local temporal motion extraction
    c1 = tf.keras.layers.Conv1D(128, kernel_size=3, padding="same", activation="relu")(x)
    c1 = tf.keras.layers.BatchNormalization()(c1)
    c1 = tf.keras.layers.Dropout(0.2)(c1)
    
    c2 = tf.keras.layers.Conv1D(128, kernel_size=5, padding="same", activation="relu")(c1)
    c2 = tf.keras.layers.BatchNormalization()(c2)
    c2 = tf.keras.layers.Dropout(0.2)(c2)
    
    # 2. Bidirectional LSTM to capture forward and backward gesture trajectory
    lstm1 = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128, return_sequences=True)
    )(c2)
    lstm1 = tf.keras.layers.Dropout(0.3)(lstm1)
    
    lstm2 = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True)
    )(lstm1)
    lstm2 = tf.keras.layers.Dropout(0.3)(lstm2)
    
    # 3. Dual Global Pooling (Avg + Max) to retain peak gesture features
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(lstm2)
    max_pool = tf.keras.layers.GlobalMaxPooling1D()(lstm2)
    pooled = tf.keras.layers.Concatenate()([avg_pool, max_pool])
    
    # 4. Dense Head with Normalization & Dropout
    d1 = tf.keras.layers.Dense(128, activation="relu")(pooled)
    d1 = tf.keras.layers.BatchNormalization()(d1)
    d1 = tf.keras.layers.Dropout(0.4)(d1)
    
    d2 = tf.keras.layers.Dense(64, activation="relu")(d1)
    d2 = tf.keras.layers.Dropout(0.3)(d2)
    
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(d2)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

model = build_model(input_shape=X.shape[1:], num_classes=num_classes)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.08),
    metrics=["accuracy"],
)

out = ROOT / "models/lstm"
out.mkdir(parents=True, exist_ok=True)
best_model_path = out / "signbridge_lstm.keras"

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(str(best_model_path), monitor="val_accuracy", save_best_only=True, verbose=1),
]

history = model.fit(
    X_train_aug[fit_idx], y_train_cat[fit_idx],
    validation_data=(X_train_aug[val_idx], y_train_cat[val_idx]),
    epochs=80, batch_size=32, callbacks=callbacks,
    class_weight=class_weight, verbose=1,
)

if best_model_path.exists():
    try:
        model = tf.keras.models.load_model(str(best_model_path))
    except Exception:
        pass

probs = model.predict(X[te], verbose=0)
pred = probs.argmax(axis=1)
model.save(str(best_model_path))
joblib.dump(enc, out / "label_encoder.joblib")

labels = list(range(num_classes))
acc = float(accuracy_score(yt[te], pred))
metrics = {
    "accuracy": acc,
    "split_type": split_type,
    "train_samples": int(len(tr)),
    "train_augmented_samples": int(len(X_train_aug)),
    "test_samples": int(len(te)),
    "classification_report": classification_report(yt[te], pred, labels=labels, target_names=enc.classes_, output_dict=True, zero_division=0),
    "classes": enc.classes_.tolist(),
    "y_true": yt[te].tolist(),
    "y_pred": pred.tolist(),
    "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
}
(out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print(f"LSTM Test Accuracy on held-out signers: {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.1f}%)")