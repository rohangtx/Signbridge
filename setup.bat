@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (
  py -3.11 -m venv venv
) else (
  python -m venv venv
)
if not exist venv\Scripts\python.exe (
  echo Could not create the virtual environment. Install Python 3.11 64-bit and retry.
  pause
  exit /b 1
)
venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 goto :error
venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto :error
echo.
echo Setup finished. Run run_pipeline_v1.bat for the V1 training pipeline.
pause
exit /b 0
:error
echo Setup failed. Check the error above, your Python version and network connection.
pause
exit /b 1
