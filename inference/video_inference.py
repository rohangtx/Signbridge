from pathlib import Path
import joblib
import tensorflow as tf
from preprocessing.landmark_extractor import LandmarkExtractor
from training.features import pooled_features, normalize_sequence

root = Path(__file__).resolve().parents[1]


def predict_video(path, model_name='lstm'):
    x = LandmarkExtractor()
    try:
        seq = x.extract_video(path)
    finally:
        x.close()
    if model_name == 'lstm':
        m = tf.keras.models.load_model(root / 'models/lstm/signbridge_lstm.keras')
        e = joblib.load(root / 'models/lstm/label_encoder.joblib')
        p = m.predict(normalize_sequence(seq)[None], verbose=0)[0]
    else:
        m = joblib.load(root / 'models/baseline/random_forest.joblib')
        e = joblib.load(root / 'models/baseline/label_encoder.joblib')
        p = m.predict_proba(pooled_features(seq)[None])[0]
    i = int(p.argmax())
    return e.inverse_transform([i])[0], float(p[i])