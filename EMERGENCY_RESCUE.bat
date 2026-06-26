@echo off
chcp 65001 >nul
echo.
echo +======================================================================+
echo +            EMERGENCY RESCUE - Inject v14 Knowledge                +
echo +======================================================================+
echo.
echo This will inject v14 knowledge into the latest v15 checkpoint
echo WITHOUT stopping current training!
echo.

cd /d D:\AI\traffic_vision\master_suite

REM Find latest v15 checkpoint
for /f "tokens=*" %%f in ('dir /b /o-n checkpoints\\v15_*.zip ^| head -1') do set LATEST=%%f

echo Latest v15 checkpoint: %LATEST%
echo.
echo Press any key to inject v14 knowledge...
pause >nul

REM Run transfer injection
call ..\venv\Scripts\activate.bat
python -c "
import sys
sys.path.insert(0, '.')
from transfer_inject import inject_v14_knowledge
inject_v14_knowledge(
    'checkpoints\\%LATEST%',
    alpha=0.6  # 60%% v14, 40%% preserved v15
)
"

echo.
echo [DONE] v14 knowledge injected!
echo [TIP] Training will automatically use new weights on next checkpoint load
echo.
pause
