"""Shared dataset, vocabulary, split and metrics helpers for sentence translation."""
from pathlib import Path
import json
import re
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split

ROOT = Path(__file__).resolve().parents[1]
START = "<start>"
END = "<end>"
PAD = "<pad>"
UNK = "<unk>"
MAX_TARGET_TOKENS = 32


def tokenize(text):
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?|[^\w\s]", str(text).lower())


_L_SHOULDER_IDX = 11
_R_SHOULDER_IDX = 12
_POSE_STRIDE = 4


def normalize_sequence(seq: np.ndarray) -> np.ndarray:
    """Center landmarks on shoulder midpoint, scale by shoulder width.
    Makes features signer-position and scale invariant.
    seq: (T, 258) float32
    """
    seq = seq.copy()
    ls = _L_SHOULDER_IDX * _POSE_STRIDE
    rs = _R_SHOULDER_IDX * _POSE_STRIDE
    for t in range(seq.shape[0]):
        lx, ly = seq[t, ls], seq[t, ls + 1]
        rx, ry = seq[t, rs], seq[t, rs + 1]
        if lx == 0.0 and rx == 0.0:
            continue
        cx = (lx + rx) / 2.0
        cy = (ly + ry) / 2.0
        shoulder_width = max(abs(rx - lx), 0.05)
        # Pose landmarks: 33 points (stride 4: x, y, z, vis)
        for i in range(0, 132, 4):
            if seq[t, i] != 0.0 or seq[t, i + 1] != 0.0:
                seq[t, i]     = (seq[t, i]     - cx) / shoulder_width
                seq[t, i + 1] = (seq[t, i + 1] - cy) / shoulder_width
        # Left hand: 132 to 195 (63 values)
        lh = seq[t, 132:195]
        if np.any(lh != 0.0):
            for i in range(132, 195, 3):
                seq[t, i]     = (seq[t, i]     - cx) / shoulder_width
                seq[t, i + 1] = (seq[t, i + 1] - cy) / shoulder_width
        # Right hand: 195 to 258 (63 values)
        rh = seq[t, 195:258]
        if np.any(rh != 0.0):
            for i in range(195, 258, 3):
                seq[t, i]     = (seq[t, i]     - cx) / shoulder_width
                seq[t, i + 1] = (seq[t, i + 1] - cy) / shoulder_width
    return seq


def normalize_dataset(X: np.ndarray) -> np.ndarray:
    """Normalize a batch of landmark sequences (N, 32, 258)."""
    normed = np.empty_like(X)
    for i in range(len(X)):
        normed[i] = normalize_sequence(X[i])
    return normed


def augment_landmarks(X_batch, jitter_std=0.008, scale_range=(0.95, 1.05)):
    """Augments landmark sequences (N, 32, 258) with subtle jitter, scaling, and frame smoothing."""
    aug = X_batch.copy()
    N, T, F = aug.shape
    scales = np.random.uniform(scale_range[0], scale_range[1], size=(N, 1, 1)).astype(np.float32)
    aug = aug * scales
    mask = (aug != 0.0)
    noise = np.random.normal(0, jitter_std, size=(N, T, F)).astype(np.float32)
    aug = np.where(mask, aug + noise, 0.0)
    for i in range(N):
        if np.random.rand() < 0.4:
            drop_idx = np.random.randint(1, T)
            aug[i, drop_idx] = aug[i, drop_idx - 1]
    return aug


def split_indices(n, groups=None, texts=None, test_size=0.15, random_state=42):
    ids = np.arange(n)
    if texts is not None:
        texts_arr = np.asarray(texts)
        from collections import Counter
        counts = Counter(texts_arr)
        multi = [c for c, count in counts.items() if count >= 2]
        if len(multi) >= 10:
            mask = np.isin(texts_arr, multi)
            multi_ids = ids[mask]
            single_ids = ids[~mask]
            tr_sub, te = train_test_split(multi_ids, test_size=test_size, random_state=random_state, stratify=texts_arr[mask])
            tr = np.concatenate([tr_sub, single_ids])
            np.random.RandomState(random_state).shuffle(tr)
            return tr, te, "stratified sentence split"
    if groups is not None:
        groups = np.asarray(groups).astype(str)
        known = groups != "unknown"
        valid = groups[known]
        if len(set(valid.tolist())) >= 2:
            for seed in range(random_state, random_state + 20):
                tr, te = next(GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed).split(ids, groups=groups))
                if len(tr) > 0 and len(te) > 0:
                    return tr, te, "signer-group split"
    tr, te = train_test_split(ids, test_size=test_size, random_state=random_state, shuffle=True)
    return tr, te, "random pair split (signer split unavailable)"


def build_vocab(texts, max_size=12000, min_freq=1):
    from collections import Counter
    count = Counter(tok for text in texts for tok in tokenize(text))
    vocab = [PAD, UNK, START, END]
    vocab += [w for w, c in count.most_common() if c >= min_freq and w not in vocab][: max(0, max_size - len(vocab))]
    return {token: idx for idx, token in enumerate(vocab)}


def encode_text(text, vocab, max_len=MAX_TARGET_TOKENS):
    ids = [vocab[START]]
    ids.extend(vocab.get(tok, vocab[UNK]) for tok in tokenize(text)[: max_len - 2])
    ids.append(vocab[END])
    ids = ids[:max_len]
    return ids + [vocab[PAD]] * (max_len - len(ids))


def decode_ids(ids, id_to_token):
    out = []
    for idx in ids:
        token = id_to_token[int(idx)] if 0 <= int(idx) < len(id_to_token) else UNK
        if token in (PAD, START):
            continue
        if token == END:
            break
        out.append(token)
    text = " ".join(out)
    return re.sub(r"\s+([.,!?;:])", r"\1", text).strip()


def load_translation_data(normalize=True):
    p = ROOT / "data" / "processed" / "translation_sequences.npz"
    if not p.exists():
        raise SystemExit("translation_sequences.npz missing. Run preprocessing/build_translation_dataset.py first.")
    d = np.load(p, allow_pickle=True)
    X = d["X"].astype("float32")
    texts = d["text"].astype(str)
    signers = d["signer"].astype(str) if "signer" in d.files else np.asarray(["unknown"] * len(texts))
    if len(X) < 10:
        raise SystemExit(f"Only {len(X)} pairs found; at least 10 are required.")
    if len(X) != len(texts):
        raise SystemExit("Feature/text count mismatch in translation_sequences.npz")
    if normalize:
        X = normalize_dataset(X)
    return X, texts, signers


def save_training_metadata(out_dir, vocab, max_len, metrics):
    out_dir.mkdir(parents=True, exist_ok=True)
    id_to_token = [None] * len(vocab)
    for token, idx in vocab.items():
        id_to_token[idx] = token
    (out_dir / "vocab.json").write_text(json.dumps({"token_to_id": vocab, "id_to_token": id_to_token, "max_target_tokens": max_len}, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def compute_text_metrics(references, hypotheses):
    exact = float(np.mean([a.strip().lower() == b.strip().lower() for a, b in zip(references, hypotheses)])) if references else 0.0
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        refs = [[tokenize(x)] for x in references]
        hyps = [tokenize(x) for x in hypotheses]
        bleu = float(corpus_bleu(refs, hyps, smoothing_function=SmoothingFunction().method1)) if refs else 0.0
    except Exception:
        bleu = None
    return {"exact_match": exact, "corpus_bleu": bleu, "evaluated_sentences": len(references), "note": "Metrics are meaningful only with a properly sized, representative held-out set; exact match and BLEU do not replace human evaluation."}
