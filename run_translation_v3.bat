@echo off
setlocal
cd /d "%~dp0"
if not exist venv\Scripts\python.exe (echo Run setup.bat first.& pause & exit /b 1)
echo [1/2] Extracting paired sentence-video landmarks...
venv\Scripts\python.exe -m preprocessing.build_translation_dataset || goto :error
echo [2/2] Training V3 encoder-decoder LSTM...
venv\Scripts\python.exe -m training.train_sentence_model || goto :error
echo V3 training completed. Check models\v3\metrics.json and reports.
pause
exit /b 0
:error
echo V3 training failed. Read the error above.
pause
exit /b 1
