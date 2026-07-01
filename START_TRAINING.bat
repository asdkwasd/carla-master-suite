@echo off
title CARLA Training v15
cd /d D:\AI\traffic_vision\master_suite
call ..\venv\Scripts\activate.bat
echo.
echo Starting CARLA Training v15 with GPU...
echo GPU: NVIDIA GeForce RTX 4080 Laptop GPU
echo.
python 1_TRAIN_MASTER_v15.py
echo.
echo Training stopped. Press any key to exit...
pause >nul
