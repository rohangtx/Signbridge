"""Autoregressive inference for the trained V3/V4 sentence models."""
from pathlib import Path
import json
import re
import numpy as np
import tensorflow as tf
from preprocessing.landmark_extractor import LandmarkExtractor

ROOT = Path(__file__).resolve().parents[1]


def _tokenize(text):
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?|[^\w\s]", str(text).lower())


def translate_video(video_path, version="v3"):
    version = version.lower()
    if version == "v3":
        folder, model_file = ROOT / "models" / "v3", "sentence_lstm.keras"
    elif version == "v4":
        folder, model_file = ROOT / "models" / "v4", "sentence_transformer.keras"
    else:
        raise ValueError("version must be 'v3' or 'v4'")
    model_path = folder / model_file
    vocab_path = folder / "vocab.json"
    if not model_path.is_file() or not vocab_path.is_file():
        raise FileNotFoundError(
            f"{version.upper()} sentence model isn't trained yet. Follow README.md to prepare paired sentence video/transcript data, then run its training batch file."
        )
    extractor = LandmarkExtractor()
    try:
        sequence = extractor.extract_video(video_path, n=32)
    finally:
        extractor.close()
    model = tf.keras.models.load_model(model_path)
    vocab_data = json.loads(vocab_path.read_text(encoding="utf-8"))
    token_to_id = vocab_data["token_to_id"]
    id_to_token = vocab_data["id_to_token"]
    max_tokens = int(vocab_data.get("max_target_tokens", 32))
    start_id = token_to_id["<start>"]
    end_id = token_to_id["<end>"]
    pad_id = token_to_id["<pad>"]
    current = [start_id]
    confidence_scores = []
    source_positions = np.arange(32, dtype=np.int32)[None, :]
    target_positions = np.arange(max_tokens - 1, dtype=np.int32)[None, :]
    for pos in range(max_tokens - 1):
        decoder = np.zeros((1, max_tokens - 1), dtype=np.int32)
        decoder[0, :min(len(current), max_tokens - 1)] = current[:max_tokens - 1]
        inputs = [sequence[None, ...], decoder]
        if version == "v4":
            inputs.extend([source_positions, target_positions])
        probs = model.predict(inputs, verbose=0)[0, pos]
        next_id = int(np.argmax(probs))
        confidence_scores.append(float(probs[next_id]))
        if next_id in (end_id, pad_id):
            break
        current.append(next_id)
    out = []
    for idx in current[1:]:
        token = id_to_token[idx] if idx < len(id_to_token) else "<unk>"
        if token in ("<pad>", "<start>", "<end>"):
            continue
        out.append(token)
    sentence = " ".join(out)
    sentence = re.sub(r"\s+([.,!?;:])", r"\1", sentence).strip()
    mean_conf = float(np.mean(confidence_scores)) if confidence_scores else 0.0
    return sentence, mean_conf
