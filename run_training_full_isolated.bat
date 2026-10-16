@echo off
setlocal
cd /d "%~dp0"
if not exist venv\Scripts\python.exe (echo Run setup.bat first. & pause & exit /b 1)
set PY=venv\Scripts\python.exe

echo ============================================================
echo  SignBridge — Full Isolated-Sign Baseline Training Pipeline
echo  Dataset: vidit031/isl-isolated-40words
echo  40 glosses, ~642 clips, ~196 MB download
echo  Trains: Random Forest + LSTM sequence classifier
echo  Disk needed: ~600 MB for data + ~200 MB for models
echo ============================================================
echo.
echo This will:
echo   1. Download ~196 MB of video data from Hugging Face.
echo   2. Run MediaPipe Holistic on all clips (may take 10-30 min).
echo   3. Train a Random Forest (fast) and an LSTM (several minutes).
echo   4. Write real accuracy metrics to reports\.
echo.
set /p CONFIRM="Press ENTER to continue or Ctrl+C to cancel: "

echo.
echo [1/8] Checking environment...
%PY% -m scripts.check_env
if errorlevel 1 (
  echo [WARN] Some dependency issues detected. Continuing — check errors above.
)

echo.
echo [2/8] Downloading 40-word isolated-sign dataset (~196 MB)...
%PY% -m scripts.download_dataset --dataset isolated-40
if errorlevel 1 goto :error

echo.
echo [3/8] Syncing dataset to data\raw (pipeline entry point)...
if not exist data\raw mkdir data\raw
copy /Y data\raw\isolated_40words\metadata.csv data\raw\metadata.csv >nul 2>&1
xcopy /E /I /Y data\raw\isolated_40words\* data\raw\ >nul 2>&1

echo.
echo [4/8] Analyzing dataset class and signer distribution...
%PY% -m training.dataset_analysis
if errorlevel 1 goto :error

echo.
echo [5/8] Extracting MediaPipe Holistic landmarks (this is the slow step)...
%PY% -m preprocessing.build_dataset
if errorlevel 1 goto :error

echo.
echo [6/8] Training Random Forest baseline classifier...
%PY% -m training.train_baseline
if errorlevel 1 goto :error

echo.
echo [7/8] Training LSTM sequence classifier...
%PY% -m training.train_lstm
if errorlevel 1 goto :error

echo.
echo [8/8] Consolidating metrics, plots and report...
%PY% -m training.evaluate
%PY% -m training.plot_metrics
%PY% -m training.model_report

echo.
echo ============================================================
echo  Full isolated-sign training completed.
echo  Results in: reports\evaluation.md
echo  Models in : models\baseline\  and  models\lstm\
echo  IMPORTANT: Accuracy is meaningful only with signer-aware
echo  splitting and a representative test split. Check reports\.
echo ============================================================
pause
exit /b 0

:error
echo.
echo Pipeline step FAILED. Read the error message above.
echo No accuracy numbers are fabricated by this script.
pause
exit /b 1
