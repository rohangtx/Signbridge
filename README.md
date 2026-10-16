# SignBridge — Unified V1–V4 Project

**Indian Sign Language (ISL) understanding and translation: isolated signs → continuous video → sentence translation.**

This repository unifies four development stages in one Python project. The interface is designed as **one product with a primary sentence-translation workflow**; V1–V4 are internal engineering milestones, not four separate user-facing products. It includes code and workflows, **not pretrained weights**. Models and metrics are created only after you obtain the appropriate dataset and train locally. No prediction, accuracy, or deployment status is fabricated.

## Development roadmap (internal)

| Version | Purpose | Included in this project | Requires training data/model? |
|---|---|---|---|
| **V1 — Isolated signs** | Classify one image or short sign clip into one of 40 starter labels; English/Hindi label display | Streamlit upload/camera-snapshot UI, MediaPipe landmarks, Random Forest baseline, LSTM sequence classifier, evaluation reports | Yes: run the 40-word dataset pipeline |
| **V2 — Continuous camera** | Run rolling 32-frame inference on a local webcam and collect stable recognized glosses | `inference/live_camera.py`, `run_live_v2.bat`, temporal smoothing/de-duplication | Yes: V1 LSTM and encoder must be trained; webcam required on the computer running the script |
| **V3 — Sentence translation** | Encode an entire sign-language video and generate an English sentence token-by-token | Manifest-driven preprocessing, encoder-decoder LSTM training and autoregressive inference | Yes: continuous ISL videos paired with sentence translations are required |
| **V4 — Transformer translation** | Research/advanced alternative using temporal Transformer attention and autoregressive text decoding | Transformer model builder, training runner and inference hook | Yes: continuous paired dataset, model training, evaluation and adequate compute |

**Important:** V2's rolling classifier is still word/gloss recognition over time, not full natural-language translation. V3/V4 are the sentence-level pathways. Hindi output in V1 is a curated vocabulary display map; sentence-level Hindi translation is a separate future language-generation step and is not falsely implied to be solved by the V1 word map.

## Recommended first setup (Windows)

1. Install **Python 3.11 64-bit**.
2. Extract the ZIP to a normal folder, e.g. `C:\Projects\SignBridge_All_Versions`.
3. Open the folder in Antigravity using **Open Folder**.
4. Double-click `setup.bat`.
5. For V1, run `run_pipeline_v1.bat` and wait for dataset extraction/training to finish.
6. For the UI, run `run.bat` and open the local URL printed in the terminal (usually `http://localhost:8501`).
7. For V2, train V1 first, then run `run_live_v2.bat` from a computer with a working camera.
8. For V3/V4, add a sentence-video translation manifest and videos as described below, then run `run_translation_v3.bat` / `run_translation_v4.bat`.

A broken laptop webcam does not stop training. You can use a phone to collect clips and transfer those clips to the project. Streamlit's browser camera control takes a snapshot; V2's continuous local webcam runner needs a working camera on the machine that runs it.

## Training V1 (40 isolated words)

`run_pipeline_v1.bat` downloads `vidit031/isl-isolated-40words` from Hugging Face, analyzes metadata, extracts 32-frame MediaPipe Holistic sequences (258 features/frame), trains Random Forest and LSTM models, and writes actual evaluation artifacts. The dataset is an aggregated research/education starter set: verify upstream provenance and license before redistribution. Network access is required to download it.

Output paths (created after running):
- `data/raw/` — downloaded dataset
- `data/processed/landmarks.npz` — extracted feature sequences
- `models/baseline/` and `models/lstm/` — trained weights/encoders/metrics
- `reports/` — class counts, evaluation reports and plots

## Preparing the V3/V4 sentence dataset

A continuous sentence translator needs paired ISL **sentence/phrase videos and matching spoken-language transcripts**, not only a 40-word label dataset. One research resource to investigate is [ISLTranslate (official repository)](https://github.com/Exploration-Lab/ISLTranslate), described by its authors as a continuous ISL-English video/sentence dataset. Follow its terms, human-validation caveats, download instructions and license/provenance notices.

1. Put sentence videos under `data/translation/videos/` (or use absolute paths in the manifest).
2. Copy `data/translation/manifest_template.csv` to `data/translation/manifest.csv`.
3. Fill one row per video: `video_path,translation,signer`. `video_path` can be relative to the manifest file or absolute. `signer` may be blank if unavailable. Do not fabricate transcripts or signer identifiers.
4. Run `run_translation_v3.bat` or `run_translation_v4.bat`.

The manifest is intentionally explicit instead of assuming that every third-party CSV uses identical column names or that every video has already been downloaded. The build step reports missing files and exits if no valid pairs are available.

V3/V4 preprocessing writes `data/processed/translation_sequences.npz`. Training writes weights, vocabulary and metrics to the corresponding `models/v3/` or `models/v4/` directory. The Streamlit app will only offer sentence inference when those real artifacts exist.

## Launch commands

```powershell
# Setup environment
setup.bat

# V1 isolated recognition pipeline
run_pipeline_v1.bat

# Web app (V1, plus V3/V4 when trained)
run.bat

# V2 local continuous-camera classifier (requires trained V1 LSTM)
run_live_v2.bat

# Sentence-level translation models (requires manifest + paired video data)
run_translation_v3.bat
run_translation_v4.bat

# Tests
venv\Scripts\python.exe -m pytest -q
```

## Antigravity workflow

Open the entire extracted repository folder, then paste `ANTIGRAVITY_MASTER_PROMPT.md` into the Agent Chat. Ask the agent to inspect and run tests before changing the architecture. Review changes before approving shell commands, installs, Git pushes or public deployment.

## Deployment notes

The Streamlit UI can be deployed after pushing the source and the **required model files** to an appropriate repository/storage location. MediaPipe + TensorFlow dependencies and trained model size may make free cloud hosting impractical. The V2 OpenCV webcam window is intended for a local machine, not a remote Streamlit server. Do not commit raw dataset/video files or SQLite history to a public repository by default. Use `.gitignore` and dataset licenses carefully.

## Product interface

The Streamlit app has four user-facing areas: **Translate a sentence**, **Recognize one sign**, **Model health**, and **About the project**. It does not ask users to choose V1/V2/V3/V4 as if they were separate complete products. The sentence workflow is the primary experience; isolated-sign recognition remains a baseline tool. Model availability is derived from real model and vocabulary files on disk. The UI avoids table conversion for prediction history and the requirements pin PyArrow 25.0.1 to remain compatible with the project’s NumPy 1.x environment.

## Limitations (by design)

- No model artifacts are included, because no real training run was performed while generating this source package.
- V1's 40-word model cannot infer arbitrary vocabulary or sentence meaning.
- V2 accumulates predicted glosses; it does not model full linguistic grammar, non-manual markers, discourse or sentence semantics.
- V3/V4 can only generalize within the coverage and quality of the paired dataset, signer diversity and evaluation protocol.
- The starter feature extractor uses pose and hand landmarks. It does not currently use full facial-expression features, which can be linguistically important in sign languages.
- Hindi sentence generation, speech output, sign avatar, multilingual/ASL support and accessibility certification are not claimed as implemented.

## Research reference

Joshi, A., Agrawal, S., & Modi, A. (2023). *ISLTranslate: Dataset for Translating Indian Sign Language*. Findings of ACL 2023. [ACL Anthology](https://aclanthology.org/2023.findings-acl.665/) · [Official dataset repository](https://github.com/Exploration-Lab/ISLTranslate)


## One-command training

After setup, run `run_all_training.bat` to train the V1 isolated-sign models, create reports, test the project, and train V3/V4 only if a valid paired video/transcript dataset is configured. V2 is the continuous-inference workflow using V1 weights, not a separate model-training job. See `ONE_CLICK_TRAINING.md` for prerequisites and skip behavior.

## Input modes: photo, video, and real-time camera

SignBridge exposes all three input types:

- **Photo:** classify one still image using the trained Random Forest model and return one vocabulary label in English or the current supported Hindi word mapping. A still image cannot represent a complete sentence.
- **Video:** in **Recognize one sign**, classify a short isolated-sign clip using the LSTM or baseline model. In **Translate a sentence**, submit a continuous clip to the V3/V4 sentence model when trained weights and vocabulary are available.
- **Real-time camera:** use the browser camera through `streamlit-webrtc`. The live preview overlays the rolling isolated-sign prediction and sequence of gloss labels when the V1 LSTM is trained. In the sentence workspace, the app also buffers a bounded recent camera clip; after stopping the stream, it can submit that recording to V3/V4 if a sentence model is actually trained.

**Important distinction:** live rolling glosses are not themselves a fluent sentence translation. Full sentence output depends on a trained model built from paired continuous ISL videos and reviewed English transcripts. Photo mode only gives isolated-sign recognition.

Camera access requires browser permission. Local development works on localhost; deployed sites generally require HTTPS and network/browser support for WebRTC. If live camera does not start, use the video upload mode or the local OpenCV runner `run_live_v2.bat`.
