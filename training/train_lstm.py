from pathlib import Path
import json
import numpy as np
import joblib
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from training.splits import split_data
from training.features import normalize_sequence  # ADDED

ROOT = Path(__file__).resolve().parents[1]
data = np.load(ROOT / "data/processed/landmarks.npz", allow_pickle=True)
X, y = data["X"].astype("float32"), data["y"]
X = np.stack([normalize_sequence(s) for s in X])  # ADDED: body-relative features, (N, T, 351)
groups = data["signer"] if "signer" in data else None
tr, te, split_type = split_data(X, y, groups)
enc = LabelEncoder(); yt = enc.fit_transform(y)
num_classes = len(enc.classes_)
weights = compute_class_weight(class_weight="balanced", classes=np.unique(yt[tr]), y=yt[tr])
class_weight = {int(c): float(w) for c, w in zip(np.unique(yt[tr]), weights)}

# FIX: Keras validation_split takes the LAST 15% of the array unshuffled. Samples are
# stored in class-folder order, so validation was whole unseen classes (val_acc = 0).
# Shuffle explicitly instead.
rng = np.random.default_rng(42)
perm = rng.permutation(tr)
n_val = max(1, int(0.15 * len(perm)))
val_idx, fit_idx = perm[:n_val], perm[n_val:]

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=X.shape[1:]),
    tf.keras.layers.Masking(mask_value=0.0),
    tf.keras.layers.LSTM(128, return_sequences=True),
    tf.keras.layers.Dropout(0.30),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dropout(0.30),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(num_classes, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
out = ROOT / "models/lstm"; out.mkdir(parents=True, exist_ok=True)
callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True)]
history = model.fit(
    X[fit_idx], yt[fit_idx],
    validation_data=(X[val_idx], yt[val_idx]),
    epochs=80, batch_size=16, callbacks=callbacks,
    class_weight=class_weight, verbose=1,
)
probs = model.predict(X[te], verbose=0); pred = probs.argmax(axis=1)
model.save(out / "signbridge_lstm.keras"); joblib.dump(enc, out / "label_encoder.joblib")
labels = list(range(num_classes))
metrics = {
    "accuracy": float(accuracy_score(yt[te], pred)), "split_type": split_type,
    "train_samples": int(len(tr)), "test_samples": int(len(te)),
    "classification_report": classification_report(yt[te], pred, labels=labels, target_names=enc.classes_, output_dict=True, zero_division=0),
    "classes": enc.classes_.tolist(), "y_true": yt[te].tolist(), "y_pred": pred.tolist(),
    "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
}
(out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print(f"LSTM accuracy: {metrics['accuracy']:.4f}")