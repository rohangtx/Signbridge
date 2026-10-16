# SignBridge landmark extraction / training fix

## Why the previous run failed

OpenCV reported `Insufficient memory` while decoding a few clips. The old preprocessing stage did not leave a verified `data/processed/landmarks.npz`, but the master runner still reached Random Forest training, which then raised `FileNotFoundError`.

## Install this patch

1. Stop SignBridge and any training terminal with `Ctrl+C`.
2. Back up `data/`, `models/`, and `reports/` if they contain anything you need.
3. Extract this add-on ZIP into the root of `A:\hehe\SignBridge_All_Versions`.
4. Choose **Replace** for the included code/script files.
5. In PowerShell, run:

```powershell
cd A:\hehe\SignBridge_All_Versions
.\run_all_training.bat
```

Do not delete `data/processed/landmark_cache` between attempts. Each successfully processed video is cached individually; if a run stops, a rerun reuses cached features.

## What the patch changes

- Samples 32 frame positions by seeking rather than decoding every frame in every video.
- Resizes frames to a maximum dimension of 640 before MediaPipe inference.
- Uses MediaPipe's lighter `model_complexity=0`, limits OpenCV threads and periodically re-creates the Holistic graph.
- Saves each successfully processed clip to an individual `.npy` cache file.
- Writes `landmarks.npz` atomically and verifies its dimensions before creating `landmarks_build_complete.json`.
- Ensures the training batch stops if extraction did not make a verified dataset.
- Adds explicit dataset checks to Random Forest training.

## If memory errors persist

Close browser tabs, Antigravity preview and memory-heavy apps before training. Keep the PC plugged in. A video that consistently fails is logged in `data/processed/failed_videos.txt` and skipped; if too few valid examples/classes remain, preprocessing stops rather than producing a misleading model.

V1 trains only the isolated 40-word vocabulary. V3/V4 sentence translation still require a valid paired video/transcript manifest and dataset.
