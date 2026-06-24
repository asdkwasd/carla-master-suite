@echo off
chcp 65001 >nul 2>&1
setlocal EnableDelayedExpansion
title CARLA GRANDMASTER AI v15 — RTX Optimized

:: ═══ CONFIG ═══
set "PROJECT_DIR=E:\AI\traffic_vision\master_suite"
set "VENV_DIR=E:\AI\traffic_vision\venv"

:: ═══ GPU OPTIMIZATION (RTX 4080 Laptop) ═══
set "NVIDIA_TF32_OVERRIDE=1"
set "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1"
set "CUDNN_BENCHMARK=1"
set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256"
set "CUDA_DEVICE_ORDER=PCI_BUS_ID"
set "CUDA_VISIBLE_DEVICES=0"
set "TORCH_CUDNN_V8_API_ENABLED=1"
set "PYTHONUNBUFFERED=1"

:: ═══ ACTIVATE VENV ═══
if exist "%VENV_DIR%\Scripts\activate.bat" (
    call "%VENV_DIR%\Scripts\activate.bat"
) else (
    echo [ERROR] venv not found: %VENV_DIR%
    pause & exit /b 1
)
cd /d "%PROJECT_DIR%"

:MENU
cls
echo.
echo  ╔═══════════════════════════════════════════════════════════════╗
echo  ║           CARLA GRANDMASTER AI v15.0 — MAIN MENU            ║
echo  ║           Tesla/Waymo Architecture + RTX Optimized           ║
echo  ╠═══════════════════════════════════════════════════════════════╣
echo  ║                                                               ║
echo  ║   [1] TRAIN          Start/resume RL training (PPO)          ║
echo  ║   [2] AI DRIVE       Watch AI drive live (60 FPS)            ║
echo  ║   [3] MANUAL DRIVE   Drive yourself + record data            ║
echo  ║   [4] COMPETITION    Multi-agent battle arena                ║
echo  ║   [5] DEPLOY         Export ONNX + TensorRT FP16             ║
echo  ║   [6] MONITOR        Open TensorBoard (browser)              ║
echo  ║   [7] GPU CHECK      Show GPU/RAM/TensorRT status            ║
echo  ║   [8] CARLA SERVER   Launch CARLA simulator                  ║
echo  ║   [0] EXIT                                                    ║
echo  ║                                                               ║
echo  ╠═══════════════════════════════════════════════════════════════╣

:: ═══ QUICK GPU STATUS (inline) ═══
for /f "tokens=1-4 delims=," %%a in ('nvidia-smi --query-gpu^=name^,memory.used^,memory.total^,temperature.gpu --format^=csv^,noheader^,nounits 2^>nul') do (
    set "GPU_NAME=%%a"
    set "VRAM_USED=%%b"
    set "VRAM_TOTAL=%%c"
    set "GPU_TEMP=%%d"
)
if defined GPU_NAME (
    echo  ║   GPU: !GPU_NAME! ^| VRAM: !VRAM_USED!/!VRAM_TOTAL! MB ^| !GPU_TEMP!C      ║
) else (
    echo  ║   GPU: [nvidia-smi not responding]                          ║
)

:: ═══ MODEL STATUS ═══
if exist "carla_grandmaster_v15.zip" (
    echo  ║   Model: carla_grandmaster_v15.zip [READY]                  ║
) else (
    echo  ║   Model: [NOT TRAINED YET]                                  ║
)
if exist "deploy\carla_v15_fp16.engine" (
    echo  ║   TensorRT: FP16 engine [READY]                             ║
) else (
    echo  ║   TensorRT: [NOT BUILT - run Deploy]                        ║
)

echo  ║   Optimizations: TF32 + cuDNN Benchmark + CUDA Pool [ON]    ║
echo  ╚═══════════════════════════════════════════════════════════════╝
echo.
set /p "c=  Select [0-8]: "

if "%c%"=="1" goto TRAIN
if "%c%"=="2" goto AI_LIVE
if "%c%"=="3" goto MANUAL
if "%c%"=="4" goto COMPETITION
if "%c%"=="5" goto DEPLOY
if "%c%"=="6" goto MONITOR
if "%c%"=="7" goto GPU_CHECK
if "%c%"=="8" goto CARLA_SERVER
if "%c%"=="0" exit /b 0
goto MENU

:: ═══════════════════════════════════════════════════════════════════
:TRAIN
cls
echo.
echo  [TRAIN] Pre-flight GPU optimization...
python gpu_optimizer.py --mode optimize --training 2>nul
echo.
echo  Starting training... (Ctrl+C = save emergency checkpoint)
echo.
python 1_TRAIN_MASTER_v15.py
echo.
echo  Auto-exporting TensorRT...
python -c "from gpu_optimizer import auto_export_tensorrt; auto_export_tensorrt()" 2>nul
pause & goto MENU

:: ═══════════════════════════════════════════════════════════════════
:AI_LIVE
cls
echo.
echo  [AI LIVE] Setting up inference...
python gpu_optimizer.py --mode optimize 2>nul
:: Auto-build TensorRT if missing
python -c "from gpu_optimizer import check_tensorrt_status, auto_export_tensorrt; s=check_tensorrt_status(); (not s['trt_fp16_exists'] and s['model_exists']) and auto_export_tensorrt()" 2>nul
echo.
echo  Controls: SPACE=override  Arrows=steer  ESC=quit
echo.
python 3_TEST_LIVE_v15.py
pause & goto MENU

:: ═══════════════════════════════════════════════════════════════════
:MANUAL
cls
echo.
echo  [MANUAL] Controls: Arrows=drive  S=save data  ESC=cancel
echo  Data saves to: human_demos\
echo.
python 4_COLLECT_HUMAN.py
pause & goto MENU

:: ═══════════════════════════════════════════════════════════════════
:COMPETITION
cls
call START_COMPETITION.bat
goto MENU

:: ═══════════════════════════════════════════════════════════════════
:DEPLOY
cls
echo.
echo  [DEPLOY] Full pipeline: Validate + ONNX + TensorRT FP16
echo.
python gpu_optimizer.py --mode optimize 2>nul
python 4_DEPLOY_MASTER_v15.py --action all
pause & goto MENU

:: ═══════════════════════════════════════════════════════════════════
:MONITOR
cls
echo  [MONITOR] Opening TensorBoard on localhost:6006...
start "" cmd /c "call "%VENV_DIR%\Scripts\activate.bat" && tensorboard --logdir "%PROJECT_DIR%\tb_logs" --port 6006"
timeout /t 3 >nul
start http://localhost:6006
echo  TensorBoard running in background. Close its terminal to stop.
pause & goto MENU

:: ═══════════════════════════════════════════════════════════════════
:GPU_CHECK
cls
echo.
python gpu_optimizer.py --mode check --training
echo.
echo  ─── nvidia-smi ───
nvidia-smi
echo.
echo  ─── Checkpoints ───
dir /b /o-d checkpoints\*.zip 2>nul | findstr /n "." | findstr /b "1: 2: 3: 4: 5:"
echo  ...
echo  Total checkpoints:
dir /b checkpoints\*.zip 2>nul | find /c /v ""
pause & goto MENU

:: ═══════════════════════════════════════════════════════════════════
:CARLA_SERVER
cls
echo.
echo  [CARLA] Select quality:
echo    [1] Low Quality (hemat RAM — RECOMMENDED)
echo    [2] Offscreen (training only, no window)
echo    [3] Normal Quality
echo.
set /p "cq=  Select [1-3]: "

:: Find CARLA
set "CARLA_EXE="
for %%d in (
    "E:\AI\traffic_vision\CARLA_0.9.16\CarlaUE4.exe"
    "C:\CARLA\CARLA_0.9.15\CarlaUE4.exe"
    "C:\CARLA\CarlaUE4.exe"
    "D:\CARLA\CARLA_0.9.15\CarlaUE4.exe"
    "D:\CARLA\CarlaUE4.exe"
) do ( if exist "%%~d" set "CARLA_EXE=%%~d" )

if not defined CARLA_EXE (
    set /p "CARLA_EXE=  CARLA path not found. Enter full path to CarlaUE4.exe: "
)

if "%cq%"=="1" start "" "%CARLA_EXE%" -quality-level=Low -prefernvidia
if "%cq%"=="2" start "" "%CARLA_EXE%" -quality-level=Low -RenderOffScreen -prefernvidia
if "%cq%"=="3" start "" "%CARLA_EXE%"

echo  Waiting 15s for CARLA to start...
timeout /t 15 >nul
echo  CARLA ready on localhost:2000
pause & goto MENU
