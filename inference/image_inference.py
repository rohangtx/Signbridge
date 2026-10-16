from pathlib import Path
import cv2
import joblib
from preprocessing.landmark_extractor import LandmarkExtractor
from training.features import pooled_features

root = Path(__file__).resolve().parents[1]


def predict_image(path):
    m = joblib.load(root / 'models/baseline/random_forest.joblib')
    e = joblib.load(root / 'models/baseline/label_encoder.joblib')
    f = cv2.imread(str(path))
    if f is None:
        raise ValueError('Could not read image')
    x = LandmarkExtractor()
    try:
        v = x.extract_frame(f)
    finally:
        x.close()
    feat = pooled_features(v[None, :])[None, :]  # single frame -> same 702-dim layout as training
    p = m.predict_proba(feat)[0]
    i = int(p.argmax())
    return e.inverse_transform([i])[0], float(p[i])