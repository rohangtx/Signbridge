"""V4: temporal Transformer encoder + autoregressive Transformer decoder for ISL video -> English."""
from pathlib import Path
import json
import numpy as np
import tensorflow as tf
from training.translation_utils import ROOT, load_translation_data, split_indices, build_vocab, encode_text, compute_text_metrics, save_training_metadata, MAX_TARGET_TOKENS, PAD, decode_ids, augment_landmarks

D_MODEL = 256
NUM_HEADS = 8
FF_DIM = 512


def transformer_block(x, name):
    a = tf.keras.layers.MultiHeadAttention(num_heads=NUM_HEADS, key_dim=D_MODEL // NUM_HEADS, dropout=0.1, name=f"{name}_self_attention")(x, x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f"{name}_norm1")(x + a)
    f = tf.keras.layers.Dense(FF_DIM, activation="relu", name=f"{name}_ff1")(x)
    f = tf.keras.layers.Dropout(0.1)(f)
    f = tf.keras.layers.Dense(D_MODEL, name=f"{name}_ff2")(f)
    return tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f"{name}_norm2")(x + f)


def build_model(n_vocab, max_target_tokens):
    video = tf.keras.Input(shape=(32, 258), name="video_features")
    target = tf.keras.Input(shape=(max_target_tokens - 1,), dtype="int32", name="decoder_tokens")
    source_positions = tf.keras.Input(shape=(32,), dtype="int32", name="source_positions")
    target_positions = tf.keras.Input(shape=(max_target_tokens - 1,), dtype="int32", name="target_positions")
    src = tf.keras.layers.Dense(D_MODEL, name="source_projection")(video)
    src_pos = tf.keras.layers.Embedding(32, D_MODEL, name="source_position_embedding")(source_positions)
    src = src + src_pos
    src = transformer_block(src, "encoder1")
    src = transformer_block(src, "encoder2")
    src = transformer_block(src, "encoder3")

    tgt = tf.keras.layers.Embedding(n_vocab, D_MODEL, mask_zero=True, name="target_embedding")(target)
    tgt_pos = tf.keras.layers.Embedding(max_target_tokens - 1, D_MODEL, name="target_position_embedding")(target_positions)
    tgt = tgt + tgt_pos
    self_attn = tf.keras.layers.MultiHeadAttention(num_heads=NUM_HEADS, key_dim=D_MODEL // NUM_HEADS, dropout=0.1, name="decoder_causal_attention")(tgt, tgt, use_causal_mask=True)
    tgt = tf.keras.layers.LayerNormalization(epsilon=1e-6)(tgt + self_attn)
    cross_attn = tf.keras.layers.MultiHeadAttention(num_heads=NUM_HEADS, key_dim=D_MODEL // NUM_HEADS, dropout=0.1, name="decoder_cross_attention")(tgt, src)
    tgt = tf.keras.layers.LayerNormalization(epsilon=1e-6)(tgt + cross_attn)
    ff = tf.keras.layers.Dense(FF_DIM, activation="relu")(tgt)
    ff = tf.keras.layers.Dropout(0.1)(ff)
    ff = tf.keras.layers.Dense(D_MODEL)(ff)
    tgt = tf.keras.layers.LayerNormalization(epsilon=1e-6)(tgt + ff)
    outputs = tf.keras.layers.Dense(n_vocab, activation="softmax", name="token_probabilities")(tgt)
    model = tf.keras.Model([video, target, source_positions, target_positions], outputs, name="SignBridgeV4_Transformer")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss="sparse_categorical_crossentropy")
    return model


def main():
    X, texts, signers = load_translation_data(normalize=True)
    tr, te, split_type = split_indices(len(X), signers, texts=texts)
    vocab = build_vocab(texts[tr])
    enc = np.stack([encode_text(t, vocab, MAX_TARGET_TOKENS) for t in texts])
    decoder_in, targets = enc[:, :-1], enc[:, 1:]
    weights = (targets != vocab[PAD]).astype("float32")

    source_positions = np.tile(np.arange(32, dtype=np.int32)[None, :], (len(X), 1))
    target_positions = np.tile(np.arange(MAX_TARGET_TOKENS - 1, dtype=np.int32)[None, :], (len(X), 1))

    # Double training data with landmark augmentation (jitter, scaling, frame smoothing)
    X_aug = augment_landmarks(X[tr])
    X_train = np.concatenate([X[tr], X_aug], axis=0)
    decoder_in_train = np.concatenate([decoder_in[tr], decoder_in[tr]], axis=0)
    targets_train = np.concatenate([targets[tr], targets[tr]], axis=0)
    weights_train = np.concatenate([weights[tr], weights[tr]], axis=0)
    src_pos_train = np.tile(np.arange(32, dtype=np.int32)[None, :], (len(X_train), 1))
    tgt_pos_train = np.tile(np.arange(MAX_TARGET_TOKENS - 1, dtype=np.int32)[None, :], (len(X_train), 1))

    model = build_model(len(vocab), MAX_TARGET_TOKENS)
    out = ROOT / "models" / "v4"
    out.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(out / "transformer_best.keras", monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-7, verbose=1),
    ]
    hist = model.fit(
        [X_train, decoder_in_train, src_pos_train, tgt_pos_train],
        targets_train,
        sample_weight=weights_train,
        validation_split=0.1,
        epochs=100,
        batch_size=8,
        callbacks=callbacks,
        verbose=1,
    )
    best_path = out / "transformer_best.keras"
    if best_path.exists():
        model = tf.keras.models.load_model(best_path)
    model.save(out / "sentence_transformer.keras")

    id_to_token = [None] * len(vocab)
    for token, idx in vocab.items():
        id_to_token[idx] = token
    refs, hyps = [], []
    for idx in te:
        refs.append(texts[idx])
        # Beam search (width=4) for high-quality evaluation
        beams = [(0.0, [vocab["<start>"]], False)]
        completed = []
        for pos in range(MAX_TARGET_TOKENS - 1):
            if not beams:
                break
            candidates = []
            for log_p, ids, fin in beams:
                if fin:
                    completed.append((log_p, ids))
                    continue
                dec = np.zeros((1, MAX_TARGET_TOKENS - 1), dtype=np.int32)
                dec[0, :min(len(ids), MAX_TARGET_TOKENS - 1)] = ids[:MAX_TARGET_TOKENS - 1]
                probs = model.predict([X[idx:idx+1], dec, source_positions[idx:idx+1], target_positions[idx:idx+1]], verbose=0)[0, pos]
                top_k = np.argsort(probs)[-4:][::-1]
                for tok in top_k:
                    tok = int(tok)
                    new_lp = log_p + float(np.log(probs[tok] + 1e-10))
                    new_ids = ids + [tok]
                    if tok in (vocab["<end>"], vocab["<pad>"]):
                        completed.append((new_lp, new_ids[:-1]))
                    else:
                        candidates.append((new_lp, new_ids, False))
            candidates.sort(key=lambda x: x[0], reverse=True)
            beams = candidates[:4]
        for log_p, ids, _ in beams:
            completed.append((log_p, ids))
        best_ids = max(completed, key=lambda x: x[0])[1] if completed else [vocab["<start>"]]
        hyps.append(decode_ids(best_ids[1:], id_to_token))

    metrics = {
        "version": "V4",
        "model": "temporal Transformer encoder-decoder",
        "split_type": split_type,
        "train_pairs": int(len(tr)),
        "test_pairs": int(len(te)),
        "vocab_size": len(vocab),
        "max_target_tokens": MAX_TARGET_TOKENS,
        "d_model": D_MODEL,
        "training_history": {k: [float(v) for v in vals] for k, vals in hist.history.items()},
        "text_metrics": compute_text_metrics(refs, hyps),
        "sample_predictions": [{"reference": r, "prediction": h} for r, h in list(zip(refs, hyps))[:20]],
    }
    save_training_metadata(out, vocab, MAX_TARGET_TOKENS, metrics)
    (out / "sample_predictions.json").write_text(json.dumps(metrics["sample_predictions"], ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in ["version", "split_type", "train_pairs", "test_pairs", "text_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
