from pathlib import Path
import numpy as np
from training.splits import split_data
from utils.labels import load_labels, display_label

def test_vocab_has_40_labels():
    assert len(load_labels()) == 40

def test_signer_split_uses_groups_when_possible():
    X = np.zeros((20, 2)); y = np.array([i % 4 for i in range(20)]); groups = np.array([f"s{i//5}" for i in range(20)])
    tr, te, kind = split_data(X, y, groups, attempts=100)
    assert "signer-aware" in kind
    assert set(groups[tr]).isdisjoint(set(groups[te]))

def test_hindi_labels():
    assert display_label("hello", "Hindi") == "नमस्ते"
    assert display_label("he", "Hindi") == "वह"
    assert display_label("she", "Hindi") == "वह"

def test_translation_vocabulary_round_trip():
    from training.translation_utils import build_vocab, encode_text, decode_ids, MAX_TARGET_TOKENS
    vocab = build_vocab(["Hello, please help me.", "Please help the student."], max_size=100)
    ids = encode_text("Hello, please help me.", vocab, MAX_TARGET_TOKENS)
    id_to_token = [None] * len(vocab)
    for token, idx in vocab.items():
        id_to_token[idx] = token
    decoded = decode_ids(ids, id_to_token)
    assert "hello" in decoded
    assert "help" in decoded


def test_version_architecture_docs_exist():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    for version in ("v1", "v2", "v3", "v4"):
        assert (root / "versions" / version / "README.md").exists()
