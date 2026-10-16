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


def split_indices(n, groups=None, test_size=0.15, random_state=42):
    ids = np.arange(n)
    if groups is not None:
        groups = np.asarray(groups).astype(str)
        known = groups != "unknown"
        valid = groups[known]
        if len(set(valid.tolist())) >= 2:
            # A group split is preferred, but don't let a malformed grouping produce an empty split.
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
    # Basic punctuation spacing cleanup; this is not a grammar-correction stage.
    return re.sub(r"\s+([.,!?;:])", r"\1", text).strip()


def load_translation_data():
    p = ROOT / "data" / "processed" / "translation_sequences.npz"
    if not p.exists():
        raise SystemExit("translation_sequences.npz missing. Run preprocessing/build_translation_dataset.py first.")
    d = np.load(p, allow_pickle=True)
    X = d["X"].astype("float32")
    texts = d["text"].astype(str)
    signers = d["signer"].astype(str) if "signer" in d.files else np.asarray(["unknown"] * len(texts))
    if len(X) < 10:
        raise SystemExit(f"Only {len(X)} pairs found; at least 10 are required to start this experimental training script. This is not a recommended research dataset size.")
    if len(X) != len(texts):
        raise SystemExit("Feature/text count mismatch in translation_sequences.npz")
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
