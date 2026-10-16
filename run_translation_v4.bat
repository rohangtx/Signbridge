@echo off
setlocal
cd /d "%~dp0"
if not exist venv\Scripts\python.exe (echo Run setup.bat first.& pause & exit /b 1)
echo [1/2] Extracting paired sentence-video landmarks...
venv\Scripts\python.exe -m preprocessing.build_translation_dataset || goto :error
echo [2/2] Training V4 temporal Transformer encoder-decoder...
venv\Scripts\python.exe -m training.train_transformer_model || goto :error
venv\Scripts\python.exe -m training.compare_translation_models
echo V4 training completed. Check models\v4\metrics.json and reports\translation_comparison.md.
pause
exit /b 0
:error
echo V4 training failed. Read the error above.
pause
exit /b 1
