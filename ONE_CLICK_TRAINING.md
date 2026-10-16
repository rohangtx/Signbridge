# One-click training

Run `run_all_training.bat` from the project folder (double-click it or enter `./run_all_training.bat` in PowerShell). If the virtual environment is missing, it invokes `setup.bat` first.

- **V1:** Downloads the 40-word dataset, extracts landmarks, trains Random Forest and LSTM, then produces evaluation artifacts.
- **V2:** Continuous-camera inference, reusing V1 weights; it is not a separate model-training stage.
- **V3:** Sentence-level LSTM encoder-decoder trained on genuine paired ISL video and English transcripts.
- **V4:** Sentence-level Transformer trained on the same paired sentence dataset.

The orchestrator skips V3/V4 if `data/translation/manifest.csv` is missing or contains fewer than 10 valid, distinct video files with non-empty transcripts. This is only a smoke-test minimum, not enough for research-quality translation claims. `manifest_template.csv` is a schema example, not training data. Review `data/translation/README.md` and `DATA_AND_LICENSE_NOTES.md` before acquiring data.

Afterward, inspect actual metrics and predictions in `models/` and `reports/`. Run `run.bat` to launch the UI. A file, UI badge, or successful training command alone is not proof of model quality; review the metrics and sample predictions.

## Runtime input options
The unified Streamlit interface supports photo upload, video upload and real-time browser-camera input. Photo/video/camera single-sign recognition requires trained V1 models. The live camera outputs rolling gloss labels; sentence translation from uploaded video or a captured camera clip requires V3/V4 trained on paired continuous ISL videos and English transcripts.
