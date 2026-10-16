# SignBridge input modes

SignBridge supports photo, uploaded video, and real-time browser camera input. These inputs do not all contain the same amount of information, so the result differs by mode.

| Input | Workspace | Output | Requirements |
|---|---|---|---|
| Photo | Recognize one sign | One predicted vocabulary label | V1 Random Forest model + encoder |
| Short video | Recognize one sign | One predicted isolated-sign label | V1 LSTM or Random Forest model |
| Uploaded continuous video | Translate a sentence | Draft English sentence | Trained V3/V4 model, vocabulary, and suitable paired training data |
| Real-time camera | Recognize one sign | Live current label plus rolling gloss sequence | Browser camera permission + trained V1 LSTM; `streamlit-webrtc` |
| Real-time camera recording | Translate a sentence | Draft English sentence after the stream stops | Captured clip + trained V3/V4 model and vocabulary |

## Setup

1. In PowerShell, change to the project folder.
2. Run `.\setup.bat` (type the command as shown below).
3. Run `.\run.bat` after setup finishes.
4. In the app, choose `Recognize one sign` for photo/video/live word classification or `Translate a sentence` for continuous video and sentence translation.

The real-time web camera uses WebRTC; allow camera permission in the browser. On public hosting, camera access generally requires HTTPS and a working WebRTC network path. Use the upload-video mode if camera permissions/network are blocked.

## Important limitations

- Photo mode cannot recover a temporal sentence from one still image; it predicts a single sign.
- The real-time V2 camera shows rolling word/gloss labels. This is not fluent sentence translation.
- Sentence output is enabled only when genuine V3/V4 model weights and vocabulary files exist. The 40-word V1 dataset is not enough to train a sentence translator.
- No camera video is written into the repository. The temporary camera buffer is used locally for the current session only.
