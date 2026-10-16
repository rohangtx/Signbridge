"""V3: train an encoder-decoder LSTM for paired ISL video features -> English sentence."""
from pathlib import Path
import json
import numpy as np
import tensorflow as tf
from training.translation_utils import ROOT, load_translation_data, split_indices, build_vocab, encode_text, compute_text_metrics, save_training_metadata, MAX_TARGET_TOKENS, PAD


def build_model(n_vocab, max_target_tokens):
    video = tf.keras.Input(shape=(32, 258), name="video_features")
    decoder_tokens = tf.keras.Input(shape=(max_target_tokens - 1,), dtype="int32", name="decoder_tokens")
    _, state_h, state_c = tf.keras.layers.LSTM(256, return_state=True, name="video_encoder_lstm")(video)
    emb = tf.keras.layers.Embedding(n_vocab, 128, mask_zero=True, name="text_embedding")(decoder_tokens)
    dec = tf.keras.layers.LSTM(256, return_sequences=True, name="text_decoder_lstm")(emb, initial_state=[state_h, state_c])
    dec = tf.keras.layers.Dropout(0.25)(dec)
    outputs = tf.keras.layers.Dense(n_vocab, activation="softmax", name="token_probabilities")(dec)
    model = tf.keras.Model([video, decoder_tokens], outputs, name="SignBridgeV3_Seq2SeqLSTM")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="sparse_categorical_crossentropy")
    return model


def main():
    X, texts, signers = load_translation_data()
    tr, te, split_type = split_indices(len(X), signers)
    # Fit the vocabulary on training transcripts only to reduce evaluation leakage.
    vocab = build_vocab(texts[tr])
    enc = np.stack([encode_text(t, vocab, MAX_TARGET_TOKENS) for t in texts])
    decoder_in, targets = enc[:, :-1], enc[:, 1:]
    weights = (targets != vocab[PAD]).astype("float32")
    model = build_model(len(vocab), MAX_TARGET_TOKENS)
    out = ROOT / "models" / "v3"
    out.mkdir(parents=True, exist_ok=True)
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True), tf.keras.callbacks.ModelCheckpoint(out / "sentence_lstm_best.keras", monitor="val_loss", save_best_only=True)]
    hist = model.fit([X[tr], decoder_in[tr]], targets[tr], sample_weight=weights[tr], validation_split=0.1, epochs=25, batch_size=8, callbacks=callbacks, verbose=1)
    best_path = out / "sentence_lstm_best.keras"
    if best_path.exists():
        model = tf.keras.models.load_model(best_path)
    model.save(out / "sentence_lstm.keras")
    id_to_token = [None] * len(vocab)
    for token, idx in vocab.items(): id_to_token[idx] = token
    refs, hyps = [], []
    for idx in te:
        refs.append(texts[idx])
        current = [vocab["<start>"]]
        for pos in range(MAX_TARGET_TOKENS - 1):
            dec = np.zeros((1, MAX_TARGET_TOKENS - 1), dtype=np.int32)
            dec[0, :len(current)] = current[:MAX_TARGET_TOKENS - 1]
            probs = model.predict([X[idx:idx+1], dec], verbose=0)[0, pos]
            nxt = int(np.argmax(probs))
            if nxt == vocab["<end>"] or nxt == vocab["<pad>"]: break
            current.append(nxt)
        from training.translation_utils import decode_ids
        hyps.append(decode_ids(current[1:], id_to_token))
    metrics = {"version": "V3", "model": "encoder-decoder LSTM", "split_type": split_type, "train_pairs": int(len(tr)), "test_pairs": int(len(te)), "vocab_size": len(vocab), "max_target_tokens": MAX_TARGET_TOKENS, "training_history": {k: [float(v) for v in vals] for k, vals in hist.history.items()}, "text_metrics": compute_text_metrics(refs, hyps), "sample_predictions": [{"reference": r, "prediction": h} for r, h in list(zip(refs, hyps))[:20]]}
    save_training_metadata(out, vocab, MAX_TARGET_TOKENS, metrics)
    (out / "sample_predictions.json").write_text(json.dumps(metrics["sample_predictions"], ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in ["version", "split_type", "train_pairs", "test_pairs", "text_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
