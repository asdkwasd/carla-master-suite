@echo off
title CARLA Full Stack Starter
echo ==========================================
echo   CARLA + Training Launcher
echo ==========================================
echo.
echo STEP 1: Starting CARLA Server...
echo   (A new window will open for CARLA)
echo   Wait ~10 seconds for CARLA to initialize
echo.
start "CARLA Server" cmd /c "E:\AI\traffic_vision\CARLA_0.9.16\CarlaUE4.exe -RenderOffScreen -quality-level=Low -carla-rpc-port=2000"
echo CARLA server starting in background...
echo.
echo STEP 2: Waiting 15 seconds for CARLA to initialize...
timeout /t 15 /nobreak >nul
echo.
echo STEP 3: Starting Training...
echo.
cd /d D:\AI\traffic_vision\master_suite
call ..\venv\Scripts\activate.bat
python 1_TRAIN_MASTER_v15.py
echo.
echo Training session ended.
pause
