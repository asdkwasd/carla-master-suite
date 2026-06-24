@echo off
REM =============================================================================
REM CARLA AI COMPETITION LAUNCHER
REM =============================================================================

echo.
echo ========================================================================
echo    CARLA AI COMPETITION LEAGUE - Main Menu
echo ========================================================================
echo.
echo [1] Start Training Competition (Multi-Agent)
echo [2] View Leaderboard
echo [3] Live Dashboard (Real-time)
echo [4] Head-to-Head Battle (1v1)
echo [5] Championship Bracket
echo [6] View Agent Details
echo [7] View Battle History
echo [8] View Memory Analysis
echo [0] Exit
echo.
set /p choice="Select option: "

if "%choice%"=="1" goto TRAIN
if "%choice%"=="2" goto LEADERBOARD
if "%choice%"=="3" goto LIVE
if "%choice%"=="4" goto BATTLE
if "%choice%"=="5" goto BRACKET
if "%choice%"=="6" goto AGENT
if "%choice%"=="7" goto BATTLES
if "%choice%"=="8" goto MEMORY
if "%choice%"=="0" goto END

:TRAIN
echo.
echo [START] Multi-Agent Training Competition
echo ----------------------------------------
echo How many agents will compete? (2-8 recommended)
set /p num_agents="Number of agents: "
echo.
echo How many training rounds? (1-10 recommended)
set /p rounds="Rounds: "
echo.
echo Steps per round? (5000-20000 recommended)
set /p steps="Steps: "
echo.
python "%~dp0\1_COMPETITION_MASTER.py" --mode train --agents %num_agents% --rounds %rounds% --steps %steps%
pause
goto MENU

:LEADERBOARD
echo.
python "%~dp0\3_COMPETITION_VIEWER.py" --mode view --top 20
pause
goto MENU

:LIVE
echo.
echo [LIVE] Starting real-time dashboard...
echo Press Ctrl+C to stop
echo.
python "%~dp0\3_COMPETITION_VIEWER.py" --mode live --refresh 3
pause
goto MENU

:BATTLE
echo.
echo [BATTLE] Head-to-Head Match
echo ----------------------------
echo.
echo Available agents:
python "%~dp0\3_COMPETITION_VIEWER.py" --mode view --top 10
echo.
set /p agent1="Enter Agent 1 ID (or press Enter for auto-select): "
set /p agent2="Enter Agent 2 ID (or press Enter for auto-select): "
python "%~dp0\2_BATTLE_ARENA.py" --mode 1v1 --agent1 "%agent1%" --agent2 "%agent2%" --steps 3000
pause
goto MENU

:BRACKET
echo.
echo [CHAMPIONSHIP] Tournament Bracket
echo -----------------------------------
python "%~dp0\2_BATTLE_ARENA.py" --mode bracket
pause
goto MENU

:AGENT
echo.
echo [AGENT] View Agent Details
echo ---------------------------
python "%~dp0\3_COMPETITION_VIEWER.py" --mode view --top 10
echo.
set /p target_agent="Enter Agent ID: "
python "%~dp0\3_COMPETITION_VIEWER.py" --mode agent --agent "%target_agent%"
pause
goto MENU

:BATTLES
echo.
python "%~dp0\3_COMPETITION_VIEWER.py" --mode battles --top 20
pause
goto MENU

:MEMORY
echo.
echo [MEMORY] Memory Analysis
echo -------------------------
python "%~dp0\3_COMPETITION_VIEWER.py" --mode view --top 10
echo.
set /p mem_agent="Enter Agent ID: "
python "%~dp0\3_COMPETITION_VIEWER.py" --mode memory --agent "%mem_agent%"
pause
goto MENU

:MENU
cls
goto START

:END
echo.
echo Thanks for using CARLA AI Competition!
echo.
exit

:START
cls
%~dp0START_COMPETITION.bat
