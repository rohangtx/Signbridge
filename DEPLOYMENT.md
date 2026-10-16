# SignBridge deployment guide

## Run locally first

1. Install Python 3.11 64-bit.
2. Extract the project and run `setup.bat`.
3. Train the model version you want to demonstrate. Do not deploy the app expecting recognition to work before its real weights exist.
4. Run `run.bat` and test the app locally with videos that were not part of the training set.

## Publish source to GitHub

1. Create a private repository first while testing.
2. From the project root, initialise Git and commit only source/config/docs. Review `git status` and `.gitignore` before each push.
3. Do not push raw videos, the full research dataset, private signers' clips, SQLite history, API keys or any data you cannot redistribute.
4. Trained `.keras`/`.joblib` files are intentionally excluded from normal Git commits. Decide on compliant model artifact hosting or Git LFS only after checking repository/model size limits and any applicable data-derived redistribution terms.

## Streamlit Community Cloud

1. Push the reviewed source branch to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/) and create an app from that repository and branch.
3. Set the main file path to `app/streamlit_app.py`.
4. Select the Python version compatible with `requirements.txt` (Python 3.11 is the intended local environment).
5. Deploy, then inspect application logs. A successful container startup alone does not mean a trained model is present or inference is correct.

## Important hosting constraints

- The application uses TensorFlow and MediaPipe; dependency installation and cold starts may be heavy on free hosted instances. Resolve build failures using the actual deployment logs rather than removing required packages at random.
- The V2 OpenCV window accesses the camera attached to the machine running `run_live_v2.bat`. It is a local workflow and should not be expected to open the deploy server's webcam from a hosted website.
- V3/V4 model weights and vocabulary files must be available to the deployed process. They are produced by training and aren't bundled into this source archive.
- The V1 photo/video/camera UI and V3/V4 video upload UI use Streamlit's browser upload/camera controls. Browser camera capture is a still frame, not a continuous stream.
- Do not use the system to make high-stakes decisions. ISL interpretation needs representative data, signer-independent evaluation, appropriate community/human review, and privacy-safe data handling.

## Pre-deployment checklist

- [ ] `python -m pytest -q` passes in the chosen environment.
- [ ] At least one chosen version's real model artifacts are available.
- [ ] Actual inference has been tested on held-out examples, including failures/unknown cases.
- [ ] Dataset and model provenance/licensing have been reviewed.
- [ ] No raw data, database files, secrets or private recordings are in Git history.
- [ ] App logs have been checked after a real hosted deployment.
