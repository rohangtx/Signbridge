from pathlib import Path
import json
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from training.splits import split_data
from training.features import pooled_features

ROOT = Path(__file__).resolve().parents[1]
data = np.load(ROOT / "data/processed/landmarks.npz", allow_pickle=True)
X, y = data["X"].astype("float32"), data["y"]
groups = data["signer"] if "signer" in data else None

features = np.stack([pooled_features(s) for s in X])
tr, te, split_type = split_data(features, y, groups)
enc = LabelEncoder(); yt = enc.fit_transform(y)
num_classes = len(enc.classes_)

model = RandomForestClassifier(n_estimators=500, n_jobs=-1, class_weight="balanced", random_state=42)
model.fit(features[tr], yt[tr])
pred = model.predict(features[te])

out = ROOT / "models/baseline"; out.mkdir(parents=True, exist_ok=True)
joblib.dump(model, out / "random_forest.joblib"); joblib.dump(enc, out / "label_encoder.joblib")
labels = list(range(num_classes))
metrics = {
    "accuracy": float(accuracy_score(yt[te], pred)), "split_type": split_type,
    "train_samples": int(len(tr)), "test_samples": int(len(te)),
    "classification_report": classification_report(yt[te], pred, labels=labels, target_names=enc.classes_, output_dict=True, zero_division=0),
    "classes": enc.classes_.tolist(), "y_true": yt[te].tolist(), "y_pred": pred.tolist(),
}
(out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print(f"Baseline accuracy: {metrics['accuracy']:.4f}")