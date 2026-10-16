@echo off
setlocal EnableExtensions
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (echo Run setup.bat first.& pause & exit /b 1)
set "PY=venv\Scripts\python.exe"
if not exist "data\processed" mkdir "data\processed"
del /q "data\processed\landmarks_build_complete.json" 2>nul

echo [1/8] Downloading isolated-word dataset...
"%PY%" -m scripts.download_isl40 || goto :error
echo [2/8] Analyzing dataset...
"%PY%" -m training.dataset_analysis || goto :error
echo [3/8] Extracting landmarks (resumable cache)...
"%PY%" -m preprocessing.build_dataset || goto :error
if not exist "data\processed\landmarks.npz" (echo landmarks.npz missing after extraction.& goto :error)
if not exist "data\processed\landmarks_build_complete.json" (echo Successful extraction marker missing.& goto :error)
"%PY%" -c "import numpy as np; p=np.load('data/processed/landmarks.npz', allow_pickle=False); print(p['X'].shape); assert p['X'].ndim==3 and p['X'].shape[1:]==(32,258) and len(p['X'])==len(p['y'])"
if errorlevel 1 goto :error
echo [4/8] Training Random Forest...
"%PY%" -m training.train_baseline || goto :error
echo [5/8] Training LSTM...
"%PY%" -m training.train_lstm || goto :error
echo [6/8] Consolidating evaluation...
"%PY%" -m training.evaluate || goto :error
echo [7/8] Creating plots...
"%PY%" -m training.plot_metrics || goto :error
echo [8/8] Writing report...
"%PY%" -m training.model_report || goto :error
echo V1 training pipeline completed. Review real results in reports\.
pause
exit /b 0
:error
echo.
echo V1 pipeline failed. Read the error above. Completed per-video cached features remain available for retry.
pause
exit /b 1
