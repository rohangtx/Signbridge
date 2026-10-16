"""Autoregressive inference for the trained V3/V4 sentence models."""
from pathlib import Path
import json
import re
import numpy as np
import tensorflow as tf
from preprocessing.landmark_extractor import LandmarkExtractor

ROOT = Path(__file__).resolve().parents[1]

from training.translation_utils import normalize_sequence

def _tokenize(text):
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?|[^\w\s]", str(text).lower())


def _greedy_decode(model, sequence, vocab, max_tokens, version):
    """Standard greedy (argmax) autoregressive decoding."""
    token_to_id = vocab["token_to_id"]
    id_to_token = vocab["id_to_token"]
    start_id = token_to_id["<start>"]
    end_id = token_to_id["<end>"]
    pad_id = token_to_id["<pad>"]
    source_positions = np.arange(32, dtype=np.int32)[None, :]
    target_positions = np.arange(max_tokens - 1, dtype=np.int32)[None, :]
    current = [start_id]
    confidence_scores = []
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
    text = " ".join(out)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text).strip()
    mean_conf = float(np.mean(confidence_scores)) if confidence_scores else 0.0
    return text, mean_conf


def _beam_decode(model, sequence, vocab, max_tokens, version, beam_width=4):
    """Beam search decoding — explores top-k candidates at each step.
    Returns the highest log-probability complete sequence.
    """
    token_to_id = vocab["token_to_id"]
    id_to_token = vocab["id_to_token"]
    start_id = token_to_id["<start>"]
    end_id = token_to_id["<end>"]
    pad_id = token_to_id["<pad>"]
    source_positions = np.arange(32, dtype=np.int32)[None, :]
    target_positions = np.arange(max_tokens - 1, dtype=np.int32)[None, :]

    # Each beam: (log_prob, token_ids_list, finished)
    beams = [(0.0, [start_id], False)]
    completed = []

    for pos in range(max_tokens - 1):
        if not beams:
            break
        candidates = []
        for log_prob, ids, finished in beams:
            if finished:
                completed.append((log_prob, ids))
                continue
            decoder = np.zeros((1, max_tokens - 1), dtype=np.int32)
            decoder[0, :min(len(ids), max_tokens - 1)] = ids[:max_tokens - 1]
            inputs = [sequence[None, ...], decoder]
            if version == "v4":
                inputs.extend([source_positions, target_positions])
            probs = model.predict(inputs, verbose=0)[0, pos]
            top_k = np.argsort(probs)[-beam_width:][::-1]
            for tok in top_k:
                tok = int(tok)
                new_log_prob = log_prob + float(np.log(probs[tok] + 1e-10))
                new_ids = ids + [tok]
                is_done = tok in (end_id, pad_id)
                if is_done:
                    completed.append((new_log_prob, new_ids[:-1]))  # drop end token
                else:
                    candidates.append((new_log_prob, new_ids, False))
        # Prune to top beam_width
        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = candidates[:beam_width]

    # Add any unfinished beams as completed
    for log_prob, ids, _ in beams:
        completed.append((log_prob, ids))

    if not completed:
        return "", 0.0

    best_log_prob, best_ids = max(completed, key=lambda x: x[0])
    out = []
    for idx in best_ids[1:]:  # skip <start>
        token = id_to_token[idx] if idx < len(id_to_token) else "<unk>"
        if token in ("<pad>", "<start>", "<end>"):
            continue
        out.append(token)
    text = " ".join(out)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text).strip()
    mean_conf = float(np.exp(best_log_prob / max(len(best_ids) - 1, 1)))
    return text, mean_conf


def translate_video(video_path, version="v4", use_beam_search=True, beam_width=4, normalize=True):
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
    if normalize:
        sequence = normalize_sequence(sequence)
    model = tf.keras.models.load_model(model_path)
    vocab_data = json.loads(vocab_path.read_text(encoding="utf-8"))
    max_tokens = int(vocab_data.get("max_target_tokens", 32))
    if use_beam_search:
        return _beam_decode(model, sequence, vocab_data, max_tokens, version, beam_width=beam_width)
    else:
        return _greedy_decode(model, sequence, vocab_data, max_tokens, version)
