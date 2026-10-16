# V1 — Isolated sign classification

Input: one image or a short isolated-sign video. Output: one vocabulary label with model probability. Uses MediaPipe Holistic 258-D features, Random Forest summary statistics and an LSTM over 32 frames.

Train using `run_pipeline_v1.bat`. Artifacts: `models/baseline/` and `models/lstm/`. A prediction is limited to classes learned during that training run.
