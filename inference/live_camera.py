"""V2 local rolling-window camera classifier. Gloss buffer is not a sentence translator."""
from collections import deque, Counter
from pathlib import Path
import argparse
import mediapipe as mp
import cv2
import joblib
import numpy as np
import tensorflow as tf
from preprocessing.landmark_extractor import LandmarkExtractor

ROOT = Path(__file__).resolve().parents[1]


def run(camera_index=0, threshold=0.40, infer_every=5):
    model_path = ROOT / "models" / "lstm" / "signbridge_lstm.keras"
    enc_path = ROOT / "models" / "lstm" / "label_encoder.joblib"
    if not model_path.is_file() or not enc_path.is_file():
        raise FileNotFoundError("V2 requires models/lstm/signbridge_lstm.keras and label_encoder.joblib. Train V1 first with run_pipeline_v1.bat.")
    model = tf.keras.models.load_model(model_path)
    encoder = joblib.load(enc_path)
    extractor = LandmarkExtractor()
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        extractor.close()
        raise RuntimeError(f"Could not open camera index {camera_index}. Check camera permission or try --camera 1.")
    frames = deque(maxlen=32)
    recent = deque(maxlen=3)
    glosses = []
    frame_count = 0
    stable_label = None
    last_added = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera frame could not be read; stopping.")
                break
            frame = cv2.flip(frame, 1)
            features = extractor.extract_frame(frame)
            frames.append(features)
            frame_count += 1
            current_text = "Warming up: 32 frames needed"
            confidence = 0.0
            if len(frames) == 32 and frame_count % max(infer_every, 1) == 0:
                probs = model.predict(np.asarray(frames, dtype=np.float32)[None, ...], verbose=0)[0]
                idx = int(np.argmax(probs))
                label = str(encoder.inverse_transform([idx])[0])
                confidence = float(probs[idx])
                recent.append((label, confidence))
                votes = Counter(x[0] for x in recent)
                candidate, count = votes.most_common(1)[0]
                if count >= 2 and np.mean([c for l, c in recent if l == candidate]) >= threshold:
                    stable_label = candidate
                    if stable_label != last_added:
                        glosses.append(stable_label.replace("_", " "))
                        glosses[:] = glosses[-20:]
                        last_added = stable_label
                current_text = f"Current window: {label.replace('_', ' ')} ({confidence:.0%})"
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 100), (20, 20, 20), -1)
            cv2.putText(frame, current_text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(frame, "Gloss buffer (not translated sentence): " + " | ".join(glosses[-7:]), (12, 67), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
            cv2.putText(frame, "Q = quit / C = clear buffer", (12, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow("SignBridge V2 — Continuous Gloss Recognition", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                glosses.clear(); last_added = None; stable_label = None
    finally:
        cap.release()
        cv2.destroyAllWindows()
        extractor.close()
    print("Recognized gloss sequence (not a natural-language translation):", " ".join(glosses))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index, usually 0")
    parser.add_argument("--threshold", type=float, default=0.40, help="Minimum mean confidence for adding glosses")
    parser.add_argument("--infer-every", type=int, default=5, help="Run model every N frames after buffer is full")
    args = parser.parse_args()
    run(args.camera, args.threshold, args.infer_every)
