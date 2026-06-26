@echo off
chcp 65001 >nul
title CARLA Grandmaster AI v15 - Transfer Learning
cd /d D:\AI\traffic_vision\master_suite

echo.
echo ╔══════════════════════════════════════════════════════════════════════════╗
echo ║          CARLA GRANDMASTER AI v15 - TRANSFER LEARNING                     ║
echo ║          Leveraging 1.76M steps from v14 for faster convergence           ║
echo ╚══════════════════════════════════════════════════════════════════════════╝
echo.

REM Check if CARLA is running
echo [CHECK] Verifying CARLA connection...
timeout /t 2 /nobreak >nul

REM Activate venv
echo [ENV] Activating virtual environment...
call ..\venv\Scripts\activate.bat

REM Run training with transfer learning
echo.
echo [START] Launching training with Transfer Learning enabled...
echo [INFO] This will automatically load v14 weights (1.76M steps) into v15
echo.

python 1_TRAIN_MASTER_v15.py

if errorlevel 1 (
    echo.
    echo [ERROR] Training failed with error code %errorlevel%
    echo [HINT] Check CARLA server is running and GPU is available
    pause
)
