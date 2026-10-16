# V3 — Sentence-level LSTM translation

Input: a continuous sign-language video plus its genuine English transcript during training. The video encoder produces a latent state and an autoregressive LSTM decoder predicts text tokens. Inference is available through the Streamlit UI once `models/v3/sentence_lstm.keras` and `models/v3/vocab.json` exist.

Prepare `data/translation/manifest.csv` with `video_path,translation,signer`; see the main README. Training uses a held-out split (signer-group split when signer data allows it) and saves exact-match/BLEU plus example predictions. These metrics must be read in context and reviewed by humans.
