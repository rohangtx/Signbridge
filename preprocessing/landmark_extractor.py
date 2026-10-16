"""Memory-conscious MediaPipe Holistic feature extraction for SignBridge.

Videos are sampled at up to 32 frame positions instead of decoding and retaining
an entire clip. Large frames are resized before inference, OpenCV threading is
limited, and the Holistic graph is periodically recreated to limit long-run RAM
pressure on Windows machines.
"""
from __future__ import annotations

import gc
from pathlib import Path

import cv2
import numpy as np

import mediapipe as mp

try:
    mp_holistic = mp.solutions.holistic
except AttributeError:
    try:
        from mediapipe.python.solutions import holistic as mp_holistic
    except Exception:
        mp_holistic = None

cv2.setNumThreads(1)

FEATURES = 258
SEQUENCE_LENGTH = 32


class LandmarkExtractor:
    def __init__(self, max_dimension: int = 640, reset_every_videos: int = 35):
        self.max_dimension = max(320, int(max_dimension))
        self.reset_every_videos = max(1, int(reset_every_videos))
        self.processed_videos = 0
        self._create_holistic()

    def _create_holistic(self) -> None:
        self.holistic = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=0,  # lower memory use for dataset preprocessing
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def _reset_holistic(self) -> None:
        try:
            self.holistic.close()
        except Exception:
            pass
        gc.collect()
        self._create_holistic()

    def _prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        largest = max(height, width)
        if largest > self.max_dimension:
            scale = self.max_dimension / float(largest)
            new_w = max(1, int(round(width * scale)))
            new_h = max(1, int(round(height * scale)))
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return frame

    def extract_frame(self, frame: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.holistic.process(rgb)
        values = []

        if result.pose_landmarks:
            for point in result.pose_landmarks.landmark:
                values.extend([point.x, point.y, point.z, point.visibility])
        else:
            values.extend([0.0] * 132)

        for hand in (result.left_hand_landmarks, result.right_hand_landmarks):
            if hand:
                for point in hand.landmark:
                    values.extend([point.x, point.y, point.z])
            else:
                values.extend([0.0] * 63)

        vector = np.asarray(values, dtype=np.float32)
        if vector.shape != (FEATURES,):
            raise ValueError(f"Unexpected landmark feature count: {vector.shape}")
        return vector

    def extract_video(self, path: str | Path, n: int = SEQUENCE_LENGTH) -> np.ndarray:
        """Return an n x 258 array. Unreadable target frames are zero-filled.

        The video capture seeks to sampled frames rather than decoding the entire
        clip. A clip with no readable sampled frames raises a clear exception.
        """
        self.processed_videos += 1
        if self.processed_videos > 1 and (self.processed_videos - 1) % self.reset_every_videos == 0:
            self._reset_holistic()

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            cap.release()
            raise ValueError("OpenCV could not open this video (corrupt/unsupported file).")

        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                raise ValueError("Video metadata has no valid frame count.")

            target_indices = np.linspace(0, max(total - 1, 0), num=n).astype(int)
            sequence = np.zeros((n, FEATURES), dtype=np.float32)
            features_by_frame: dict[int, np.ndarray] = {}
            failures = 0

            for frame_index in np.unique(target_indices):
                try:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                    ok, frame = cap.read()
                    if not ok or frame is None or frame.size == 0:
                        failures += 1
                        continue
                    frame = self._prepare_frame(frame)
                    features_by_frame[int(frame_index)] = self.extract_frame(frame)
                    del frame
                except (cv2.error, MemoryError) as exc:
                    failures += 1
                    print(f"  Frame {frame_index} in {Path(path).name} skipped: {exc}")
                    gc.collect()

            for position, frame_index in enumerate(target_indices):
                feature = features_by_frame.get(int(frame_index))
                if feature is not None:
                    sequence[position] = feature

            if not features_by_frame:
                raise ValueError(
                    f"No sampled frames could be decoded ({failures} failed). "
                    "Try a smaller/valid MP4 clip or free system memory."
                )
            return sequence
        finally:
            cap.release()
            gc.collect()

    def close(self) -> None:
        try:
            self.holistic.close()
        finally:
            gc.collect()
