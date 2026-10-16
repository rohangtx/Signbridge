# V4 — Temporal Transformer translation

This advanced comparison uses a temporal Transformer encoder for frame-landmark features, plus causal self-attention and cross-attention for autoregressive English token generation. The training entry point is `training/train_transformer_model.py` and the launch wrapper is `run_translation_v4.bat`.

The Transformer is not automatically better than V3; performance must be measured on the same held-out paired dataset and preferably signer-independent test data. The source package does not include pretrained weights.
