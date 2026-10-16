# SignBridge V1–V4: Antigravity Master Prompt

You are a senior computer-vision and machine-learning engineer. Work inside this repository and evolve SignBridge from isolated ISL sign classification into continuous sign-language sentence translation.

## Product goal
Build a credible research/college project with four explicit stages:
- V1: isolated 40-word classification from an image or short video.
- V2: rolling 32-frame local-camera inference with temporal smoothing and a visible recognized-gloss buffer.
- V3: sentence-level English translation using an encoder-decoder sequence model trained on paired continuous ISL videos and sentence transcripts.
- V4: Transformer-based temporal encoder/decoder as an advanced comparison model.

## Product and UI requirement
- The public interface must feel like one coherent, polished product, not a developer dashboard. Do not display V1/V2/V3/V4 as four equal ready-made products or claim readiness from source code alone. Use plain-language workflows such as “Translate a sentence,” “Recognize one sign,” and “Model health.” Keep version numbers and architecture names in technical documentation/diagnostics, not in the primary navigation.
- Sentence-level ISL-to-English translation is the intended primary workflow. Single-sign recognition is explicitly a limited baseline. Make missing training data and missing model artifacts clear without fabricated sample predictions.
- Keep all three user input types available: photo, uploaded video, and real-time browser camera. Photo mode outputs one isolated-sign label; uploaded video can use isolated-sign or sentence translation depending on the selected workspace and trained model; real-time camera shows live rolling sign labels, and captured camera video may be submitted to the sentence engine only when genuine V3/V4 weights exist. Never label rolling glosses as a fluent translation.
- Use a restrained, production-quality visual system, accessible contrast, responsive layout, clear upload guidance, loading/error states, an intentional empty state, and a polished main work area. The UI should resemble a focused modern AI product, not a beginner Streamlit demo: establish strong typographic hierarchy, consistent spacing, custom brand treatment, clear upload/output panels, subtle interaction feedback, and a dedicated diagnostics area. Avoid cluttered version cards, emoji-only navigation, generic dashboards, false “Ready” badges, or claims that weights exist when they do not. Read `FRONTEND_DESIGN_SPEC.md` and match it. Keep the primary workflow extremely obvious.

## Non-negotiable rules
1. Inspect the existing files and actual data format before editing.
2. Do not fabricate a model, prediction, metric, training result, dataset license or deployment success.
3. Do not label a concatenation of predicted words as a verified natural-language translation. Clearly distinguish V2 gloss accumulation from V3/V4 sentence translation.
4. For sentence training, require genuine paired video/transcript data. Never derive sentence transcripts from the V1 word vocabulary or make up labels.
5. Preserve upstream data provenance, license and signer metadata. Do not invent signer identities.
6. Add robust paths, errors, tests and useful documentation. Run tests and startup checks after changes and report exactly what ran.
7. Never publish to GitHub or deploy publicly without explicit approval.
8. Do not store API keys, secrets, private videos or user camera content in the repository.

## Required work
- Audit/repair V1 dataset download, preprocessing, signer-aware splitting, Random Forest/LSTM training, metrics and plots.
- Verify `training/evaluate.py` exists and that every batch command references real scripts.
- Provide V2 real-time camera inference inside the Streamlit app via `streamlit-webrtc`, with a clear browser permission/network fallback; keep `run_live_v2.bat` as an optional native OpenCV route. The rolling camera engine requires a trained V1 LSTM model and must state that its gloss sequence is not a sentence translation.
- Verify the V3 manifest schema (`video_path,translation,signer`), paired-video preprocessing, vocabulary, teacher-forced training and autoregressive inference.
- Verify V4 Transformer inputs/outputs, masking, autoregressive decoding and evaluation.
- Make the Streamlit app display model readiness and never run random/placeholder inference when models are absent.
- Run syntax checks and pytest. If expensive training or network data download is unavailable, report it as not tested rather than pretending it succeeded.
- Finish with a concise report of changed files, tests actually run, missing data/model files, commands to execute next, and remaining limitations.
