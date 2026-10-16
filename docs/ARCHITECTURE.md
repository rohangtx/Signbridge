# Architecture and version boundaries

```text
Video/image input
    |
    +-- V1: image / isolated clip
    |       -> MediaPipe Holistic
    |       -> 258 features per frame
    |       -> Random Forest summary baseline / LSTM sequence classifier
    |       -> one class label
    |
    +-- V2: local camera stream
    |       -> rolling 32-frame buffer
    |       -> V1 LSTM repeated inference
    |       -> smoothing + stable gloss buffer
    |       -> NOT sentence translation
    |
    +-- V3: full continuous clip
    |       -> 32-frame sampled landmark sequence
    |       -> LSTM encoder-decoder trained on paired text
    |       -> autoregressive English sentence
    |
    +-- V4: full continuous clip
            -> projected temporal landmarks + Transformer encoder
            -> causal Transformer decoder + cross-attention
            -> autoregressive English sentence
```

## Landmark input schema
- Pose: 33 landmarks × (x, y, z, visibility) = 132 values.
- Left hand: 21 landmarks × (x, y, z) = 63 values.
- Right hand: 21 landmarks × (x, y, z) = 63 values.
- Total: 258 values per sampled frame.

This starter extractor does not include face landmarks or a dedicated gloss alignment module. Both may matter for accurate sign-language interpretation and are research extensions, not silently assumed capabilities.
