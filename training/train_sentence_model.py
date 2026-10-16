"""V3: train an encoder-decoder LSTM for paired ISL video features -> English sentence."""
from pathlib import Path
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
from training.translation_utils import ROOT, load_translation_data, split_indices, build_vocab, encode_text, compute_text_metrics, save_training_metadata, MAX_TARGET_TOKENS, PAD, augment_landmarks, decode_ids

LSTM_UNITS = 384
EMBED_DIM = 192
L2_REG = 1e-4


def build_model(n_vocab, max_target_tokens):
    video = tf.keras.Input(shape=(32, 258), name="video_features")
    decoder_tokens = tf.keras.Input(shape=(max_target_tokens - 1,), dtype="int32", name="decoder_tokens")
    # Video encoder: bidirectional LSTM to capture richer temporal context
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(LSTM_UNITS // 2, return_state=False, dropout=0.2,
                             kernel_regularizer=regularizers.l2(L2_REG)),
        name="video_encoder_bilstm"
    )(video)
    state_h = tf.keras.layers.Dense(LSTM_UNITS, activation="tanh", name="bridge_h")(x)
    state_c = tf.keras.layers.Dense(LSTM_UNITS, activation="tanh", name="bridge_c")(x)
    emb = tf.keras.layers.Embedding(n_vocab, EMBED_DIM, mask_zero=True, name="text_embedding")(decoder_tokens)
    dec = tf.keras.layers.LSTM(LSTM_UNITS, return_sequences=True, dropout=0.2,
                               kernel_regularizer=regularizers.l2(L2_REG),
                               name="text_decoder_lstm")(emb, initial_state=[state_h, state_c])
    dec = tf.keras.layers.Dropout(0.3)(dec)
    outputs = tf.keras.layers.Dense(n_vocab, activation="softmax", name="token_probabilities")(dec)
    model = tf.keras.Model([video, decoder_tokens], outputs, name="SignBridgeV3_Seq2SeqLSTM")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4), loss="sparse_categorical_crossentropy")
    return model


def main():
    X, texts, signers = load_translation_data(normalize=True)
    tr, te, split_type = split_indices(len(X), signers, texts=texts)
    # Fit the vocabulary on training transcripts only
    vocab = build_vocab(texts[tr])
    enc = np.stack([encode_text(t, vocab, MAX_TARGET_TOKENS) for t in texts])
    decoder_in, targets = enc[:, :-1], enc[:, 1:]
    weights = (targets != vocab[PAD]).astype("float32")

    # Double training data with landmark augmentation (jitter, scaling, frame smoothing)
    X_aug = augment_landmarks(X[tr])
    X_train = np.concatenate([X[tr], X_aug], axis=0)
    decoder_in_train = np.concatenate([decoder_in[tr], decoder_in[tr]], axis=0)
    targets_train = np.concatenate([targets[tr], targets[tr]], axis=0)
    weights_train = np.concatenate([weights[tr], weights[tr]], axis=0)

    model = build_model(len(vocab), MAX_TARGET_TOKENS)
    out = ROOT / "models" / "v3"
    out.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(out / "sentence_lstm_best.keras", monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
    ]
    hist = model.fit(
        [X_train, decoder_in_train],
        targets_train,
        sample_weight=weights_train,
        validation_split=0.1,
        epochs=80,
        batch_size=8,
        callbacks=callbacks,
        verbose=1,
    )
    best_path = out / "sentence_lstm_best.keras"
    if best_path.exists():
        model = tf.keras.models.load_model(best_path)
    model.save(out / "sentence_lstm.keras")

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
                probs = model.predict([X[idx:idx+1], dec], verbose=0)[0, pos]
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
        "version": "V3",
        "model": "encoder-decoder LSTM",
        "split_type": split_type,
        "train_pairs": int(len(tr)),
        "test_pairs": int(len(te)),
        "vocab_size": len(vocab),
        "max_target_tokens": MAX_TARGET_TOKENS,
        "training_history": {k: [float(v) for v in vals] for k, vals in hist.history.items()},
        "text_metrics": compute_text_metrics(refs, hyps),
        "sample_predictions": [{"reference": r, "prediction": h} for r, h in list(zip(refs, hyps))[:20]],
    }
    save_training_metadata(out, vocab, MAX_TARGET_TOKENS, metrics)
    (out / "sample_predictions.json").write_text(json.dumps(metrics["sample_predictions"], ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in ["version", "split_type", "train_pairs", "test_pairs", "text_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
