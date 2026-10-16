# SignBridge sentence-translation comparison

| Version | Model | Split | Train pairs | Test pairs | Exact-match | Corpus BLEU |
|---|---|---|---:|---:|---:|---:|
| V3 | LSTM encoder-decoder | signer-group split | 588 | 54 | 0.0556 | 0.0311 |
| V4 | Temporal Transformer | signer-group split | 588 | 54 | 0.2407 | 0.0759 |

## Interpretation

- These values are generated from saved held-out predictions; no scores are hard-coded.
- Compare models only when trained on the same paired dataset and evaluation split.
- Automatic metrics are not a substitute for fluent-signer/human assessment of meaning, grammar, omissions and hallucinations.
- Very small datasets or transcript noise can make the metrics unstable.