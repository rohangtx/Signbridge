# Continuous ISL translation data

This is a schema template only. `manifest_template.csv` contains one illustrative row; **it is not a real training example and must not be used for training**. Copy it to `manifest.csv` and replace the row with actual licensed video/transcript pairs. Do not leave the example row in the manifest.

Required fields:
- `video_path`: video path relative to this manifest or an absolute path.
- `translation`: verified English sentence/phrase corresponding to that exact clip.
- `signer`: true signer ID if provided; blank is permitted if unknown.

V3/V4 training is not possible until real paired videos and transcripts exist. See the official [ISLTranslate repository](https://github.com/Exploration-Lab/ISLTranslate) for dataset acquisition and terms. Confirm local CSV column names and map them into this project's manifest rather than assuming file naming.
