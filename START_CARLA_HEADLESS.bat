@echo off
REM ============================================================
REM  CARLA HEADLESS LAUNCHER for RTX 4080 GPU Optimization
REM ============================================================
REM  This script launches CARLA in headless mode to free up
REM  the RTX 4080 GPU for neural network training.
REM
REM  Usage: Double-click this file OR run from command line
REM ============================================================

setlocal enabledelayedexpansion

REM === Configuration ===
set "CARLA_ROOT=D:\AI\traffic_vision\CARLA_0.9.16"
set "MAP= Town03 "
set "PORT=2000"
set "TM_PORT=8000"

echo.
echo ============================================================
echo   CARLA HEADLESS MODE - GPU Optimization
echo ============================================================
echo   CARLA Root: %CARLA_ROOT%
echo   Map: %MAP%
echo   Port: %PORT%
echo   Traffic Manager Port: %TM_PORT%
echo ============================================================
echo.

REM === Kill any existing CARLA processes ===
echo [CLEANUP] Stopping existing CARLA processes...
taskkill /F /IM CarlaUE4* /T >nul 2>&1
timeout /t 2 /nobreak >nul

REM === Check for CARLA binary ===
if not exist "%CARLA_ROOT%\CarlaUE4.exe" (
    echo [ERROR] CarlaUE4.exe not found at %CARLA_ROOT%
    echo Please update CARLA_ROOT path in this script
    pause
    exit /b 1
)

REM === Launch CARLA in headless mode ===
echo [START] Launching CARLA in headless mode...
echo.

cd /d "%CARLA_ROOT%"

REM Headless flags:
REM   -RenderOffScreen          : No window rendering
REM   -NoGraphicsClient         : Disable graphics client
REM   -carla-server             : Server mode
REM   -world-port=%PORT%        : Network port
REM   -TrafficManagerPort=%TM_PORT% : TM port
REM   -quality-level=Low        : Minimal quality (for any rendering)
REM   -carla-no-hud             : No HUD

start "" /B "CarlaUE4.exe" ^
    -RenderOffScreen ^
    -NoGraphicsClient ^
    -carla-server ^
    -world-port=%PORT% ^
    -TrafficManagerPort=%TM_PORT% ^
    -quality-level=Low ^
    -carla-no-hud ^
    -carla-no-hud ^
    %MAP%

echo [OK] CARLA server started in headless mode
echo [OK] Listening on port %PORT%
echo.
echo ============================================================
echo   CARLA is running. You can now start training with:
echo   python 1_TRAIN_MASTER_v15.py
echo   OR
echo   python 2_TRAIN_WITH_HEADLESS_CARLA.py
echo ============================================================
echo.

REM === Wait for CARLA to be ready ===
echo [WAIT] Testing CARLA connection...

set "CARLA_READY=0"
set "MAX_RETRIES=30"
set "RETRY_COUNT=0"

:wait_loop
python -c "import carla; client=carla.Client('localhost',%PORT%); client.set_timeout(5); world=client.get_world(); print('[OK] CARLA ready:', world.get_map().name)" >nul 2>&1
if !errorlevel! equ 0 (
    set "CARLA_READY=1"
    goto :carla_ready
)

set /a RETRY_COUNT+=1
if !RETRY_COUNT! lss %MAX_RETRIES% (
    timeout /t 2 /nobreak >nul
    goto :wait_loop
)

:carla_ready
if !CARLA_READY! equ 1 (
    echo [OK] CARLA is ready!
) else (
    echo [WARN] CARLA may still be starting...
    echo [WARN] You can check with nvidia-smi
)

echo.
echo [INFO] CARLA process will continue running.
echo [INFO] Close this window to stop CARLA.
echo.

endlocal
