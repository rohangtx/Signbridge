"""Browser live-camera processor for SignBridge.

The rolling classifier predicts isolated-word glosses. Its output is not a
natural-language sentence translation. A bounded JPEG buffer can be exported
as a clip for the sentence model after capture stops.
"""
from collections import Counter, deque
from pathlib import Path
import threading

import cv2
import numpy as np

try:
    from streamlit_webrtc import VideoProcessorBase
except ImportError:  # Keep offline tests/imports possible before setup runs.
    VideoProcessorBase = object

ROOT = Path(__file__).resolve().parents[1]


class LiveSignProcessor(VideoProcessorBase):
    """Stateful video processor used by streamlit-webrtc's VideoProcessorBase."""

    def __init__(self, enable_recording=True, camera_max_width=640, max_recorded_frames=450):
        self.enable_recording = enable_recording
        self.camera_max_width = int(camera_max_width)
        self.max_recorded_frames = int(max_recorded_frames)
        self._lock = threading.Lock()
        self.frame_count = 0
        self.frames = deque(maxlen=32)
        self.recent = deque(maxlen=3)
        self.recorded_jpegs = deque(maxlen=self.max_recorded_frames)
        self.glosses = []
        self.latest_label = "Waiting for 32 frames"
        self.latest_confidence = 0.0
        self.stable_label = None
        self.last_added = None
        self.model_error = ""
        self.model = None
        self.encoder = None
        self.extractor = None
        self._load_classifier()

    def _load_classifier(self):
        model_path = ROOT / "models" / "lstm" / "signbridge_lstm.keras"
        encoder_path = ROOT / "models" / "lstm" / "label_encoder.joblib"
        if not model_path.is_file() or not encoder_path.is_file():
            self.model_error = "Live sign classification needs the trained V1 LSTM model. Train V1 first."
            return
        try:
            import joblib
            import tensorflow as tf
            from preprocessing.landmark_extractor import LandmarkExtractor
            self.model = tf.keras.models.load_model(model_path)
            self.encoder = joblib.load(encoder_path)
            self.extractor = LandmarkExtractor()
            self.model_error = ""
        except Exception as exc:
            self.model_error = f"Could not load live model: {exc}"
            self.model = self.encoder = self.extractor = None

    def recv(self, frame):
        """Process an av.VideoFrame and return an annotated frame."""
        import av

        image = frame.to_ndarray(format="bgr24")
        if image.shape[1] > self.camera_max_width:
            new_height = max(1, int(image.shape[0] * self.camera_max_width / image.shape[1]))
            image = cv2.resize(image, (self.camera_max_width, new_height), interpolation=cv2.INTER_AREA)

        self.frame_count += 1
        if self.enable_recording and self.frame_count % 3 == 0:
            ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                with self._lock:
                    self.recorded_jpegs.append(encoded.tobytes())

        if self.extractor is not None and self.model is not None and self.encoder is not None:
            try:
                features = self.extractor.extract_frame(image)
                self.frames.append(features)
                if len(self.frames) == 32 and self.frame_count % 5 == 0:
                    probabilities = self.model.predict(np.asarray(self.frames, dtype=np.float32)[None, ...], verbose=0)[0]
                    index = int(np.argmax(probabilities))
                    label = str(self.encoder.inverse_transform([index])[0])
                    confidence = float(probabilities[index])
                    self.recent.append((label, confidence))
                    votes = Counter(name for name, _ in self.recent)
                    candidate, votes_count = votes.most_common(1)[0]
                    candidate_scores = [score for name, score in self.recent if name == candidate]
                    if votes_count >= 2 and candidate_scores and float(np.mean(candidate_scores)) >= 0.40:
                        with self._lock:
                            self.stable_label = candidate
                            if candidate != self.last_added:
                                self.glosses.append(candidate.replace("_", " "))
                                self.glosses[:] = self.glosses[-30:]
                                self.last_added = candidate
                            self.latest_label = label.replace("_", " ")
                            self.latest_confidence = confidence
            except Exception as exc:
                self.model_error = f"Live inference failed: {exc}"

        # Annotate the camera preview with actual live inference state.
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (overlay.shape[1], min(108, overlay.shape[0])), (18, 27, 48), -1)
        cv2.putText(overlay, f"SignBridge LIVE | {self.latest_label[:52]}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(overlay, f"Score: {self.latest_confidence:.0%} | Rolling sign labels, not sentence translation", (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (185, 225, 238), 1, cv2.LINE_AA)
        gloss_text = "Sequence: " + " | ".join(self.glosses[-7:])
        cv2.putText(overlay, gloss_text[:105], (12, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 240, 206), 1, cv2.LINE_AA)
        return av.VideoFrame.from_ndarray(overlay, format="bgr24")

    def snapshot(self):
        with self._lock:
            return {
                "latest_label": self.latest_label,
                "confidence": float(self.latest_confidence),
                "glosses": list(self.glosses),
                "frame_count": int(self.frame_count),
                "recorded_frames": len(self.recorded_jpegs),
                "model_error": self.model_error,
            }

    def export_recording(self, destination, fps=10.0):
        """Export the bounded camera capture buffer to an MP4 file."""
        destination = Path(destination)
        with self._lock:
            payloads = list(self.recorded_jpegs)
        if len(payloads) < 8:
            raise ValueError("Not enough camera frames captured yet. Record for at least a few seconds first.")

        images = []
        for payload in payloads:
            image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is not None:
                images.append(image)
        if len(images) < 8:
            raise ValueError("The captured camera frames could not be decoded. Please try recording again.")

        height, width = images[0].shape[:2]
        writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
        if not writer.isOpened():
            raise RuntimeError("Could not create the camera recording file (MP4 writer unavailable).")
        try:
            for image in images:
                if image.shape[:2] != (height, width):
                    image = cv2.resize(image, (width, height))
                writer.write(image)
        finally:
            writer.release()
        return destination

    def close(self):
        if self.extractor is not None:
            try:
                self.extractor.close()
            except Exception:
                pass
            self.extractor = None
