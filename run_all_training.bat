@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title SignBridge - One-click training (memory-safe)

if not exist "venv\Scripts\python.exe" (
    echo [SETUP] Virtual environment not found. Running setup.bat...
    call setup.bat
    if errorlevel 1 goto :fatal
)
set "PY=venv\Scripts\python.exe"
if not exist "%PY%" goto :fatal

if not exist "data\processed" mkdir "data\processed"
del /q "data\processed\landmarks_build_complete.json" 2>nul

echo ============================================================
echo SignBridge one-click training workflow
echo ============================================================
echo Video extraction uses resized frames and per-video cache.
echo If interrupted, completed videos will be reused on the next run.
echo V3/V4 require a valid paired sentence dataset.
echo.

echo [1/9] Downloading V1 dataset...
"%PY%" -m scripts.download_isl40
if errorlevel 1 goto :v1_fatal

echo [2/9] Analyzing V1 dataset...
"%PY%" -m training.dataset_analysis
if errorlevel 1 goto :v1_fatal

echo [3/9] Extracting V1 landmarks (resumable cache)...
"%PY%" -m preprocessing.build_dataset
if errorlevel 1 goto :v1_fatal
if not exist "data\processed\landmarks.npz" (
    echo [FAILED] Landmark extraction returned without creating landmarks.npz.
    goto :v1_fatal
)
if not exist "data\processed\landmarks_build_complete.json" (
    echo [FAILED] No successful-build marker was written. Refusing to train on partial/stale data.
    goto :v1_fatal
)
"%PY%" -c "import numpy as np; p=np.load('data/processed/landmarks.npz', allow_pickle=False); print('Verified X shape:', p['X'].shape, '| samples:', len(p['y']), '| classes:', len(set(p['y'].tolist()))); assert p['X'].ndim==3 and p['X'].shape[1:]==(32,258) and len(p['X'])==len(p['y']) and len(set(p['y'].tolist()))>=2"
if errorlevel 1 goto :v1_fatal

echo [4/9] Training V1 Random Forest...
"%PY%" -m training.train_baseline
if errorlevel 1 goto :v1_fatal

echo [5/9] Training V1 LSTM...
"%PY%" -m training.train_lstm
if errorlevel 1 goto :v1_fatal

echo [6/9] Generating V1 evaluation and reports...
"%PY%" -m training.evaluate
if errorlevel 1 goto :v1_fatal
"%PY%" -m training.plot_metrics
if errorlevel 1 echo [WARNING] Plot generation failed; continuing.
"%PY%" -m training.model_report
if errorlevel 1 echo [WARNING] Report generation failed; continuing.

echo.
echo [7/9] Checking whether V3/V4 sentence data exists...
"%PY%" -m scripts.check_translation_manifest
if errorlevel 3 goto :translation_skip
if errorlevel 2 goto :translation_skip

echo [8/9] Extracting sentence-level video features...
"%PY%" -m preprocessing.build_translation_dataset
if errorlevel 1 goto :translation_skip

echo [V3] Training sentence-level LSTM encoder-decoder...
"%PY%" -m training.train_sentence_model
if errorlevel 1 (echo [WARNING] V3 failed; continuing to V4.) else (echo [OK] V3 command completed; verify models\v3\metrics.json.)

echo [V4] Training sentence-level Transformer...
"%PY%" -m training.train_transformer_model
if errorlevel 1 (echo [WARNING] V4 failed.) else (echo [OK] V4 command completed; verify models\v4\metrics.json.)
"%PY%" -m training.compare_translation_models
if errorlevel 1 echo [WARNING] Translation comparison was not generated.
goto :post_training

:translation_skip
echo.
echo [SKIPPED] V3/V4: valid paired video/transcript data is not configured or preprocessing failed.
echo V1 results remain available if V1 completed. V2 is continuous inference using V1 weights.

goto :post_training

:post_training
echo.
echo [9/9] Running project tests...
"%PY%" -m pytest -q tests
if errorlevel 1 (echo [WARNING] Some tests failed; inspect the output.) else (echo [OK] Tests passed.)
echo.
echo Workflow finished. Review actual artifacts in models\ and reports\.
echo Launch the app separately using run.bat.
echo Do not treat missing artifacts or UI badges as proof of trained models.
pause
exit /b 0

:v1_fatal
echo.
echo [FAILED] V1 download, preprocessing or training failed. Read the error above.
echo The per-video cache remains in data\processed\landmark_cache; fix the problem and rerun this script.
goto :end_failure

:fatal
echo.
echo [FAILED] Setup failed or venv\Scripts\python.exe is missing.
:end_failure
pause
exit /b 1
