# SignBridge development milestones (internal roadmap)

SignBridge is **one product**, not four separate apps. The labels V1–V4 are engineering milestones used in documentation and training scripts; they should not be presented as four ready-to-use products in the user interface.

- **Foundation — isolated sign recognition:** classify a single sign image or short clip using the starter vocabulary.
- **Continuous input — gloss stream:** analyse successive camera frames and stabilise isolated-label predictions. A gloss stream is not a sentence translation.
- **Sentence translation — sequence model:** encode a continuous ISL video and generate an English sentence. Requires paired video/transcript data and a trained model.
- **Advanced translation — Transformer:** an experimental alternative to compare with the sequence model. Requires the same kind of paired data and independent evaluation.

## Start here on Windows

1. Install Python 3.11 64-bit.
2. Run `setup.bat` once.
3. Run `run.bat` to open the unified SignBridge interface.
4. Use **Translate a sentence** for the intended end goal. If no sentence model exists yet, the app will explain what data/training is missing.
5. Use **Recognize one sign** only for the isolated-word baseline.
6. Run `run_pipeline_v1.bat` to train the initial 40-label baseline. This does not train the sentence translator.
7. Prepare paired sentence videos and transcripts before running `run_translation_v3.bat` or `run_translation_v4.bat`.

The interface shows capabilities based on actual model artifacts on disk; source code or a model architecture alone is not considered a trained model.
