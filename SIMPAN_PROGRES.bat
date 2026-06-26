@echo off
chcp 65001 >nul
title Simpan Progres CARLA ke GitHub
setlocal

REM ================== KONFIGURASI ==================
set "PROJECT_DIR=D:\AI\traffic_vision\master_suite"
set "REPO_NAME=carla-master-suite"
REM ================================================

echo.
echo ============================================================
echo    SIMPAN PROGRES CARLA  (otomatis, tidak perlu ngetik)
echo ============================================================
echo.

cd /d "%PROJECT_DIR%"
if errorlevel 1 (
  echo [GAGAL] Folder tidak ditemukan: %PROJECT_DIR%
  echo Buka file ini dengan Notepad dan perbaiki baris PROJECT_DIR.
  pause
  exit /b 1
)

where git >nul 2>&1
if errorlevel 1 (
  echo [GAGAL] Git belum terpasang.
  echo Pasang dulu dari https://git-scm.com lalu jalankan file ini lagi.
  pause
  exit /b 1
)

if not exist ".git" (
  echo [1/5] Membuat repo lokal baru...
  git init
  git branch -M main
) else (
  echo [1/5] Repo lokal sudah ada.
)

echo [2/5] Menyiapkan daftar file yang diabaikan...
> .gitignore (
  echo checkpoints/
  echo *.zip
  echo *.pth
  echo *.log
  echo __pycache__/
  echo *.pyc
  echo tb_logs/
)

echo [3/5] Membersihkan file besar/sampah dari pelacakan...
git rm -r --cached --ignore-unmatch checkpoints __pycache__ core/__pycache__ tb_logs >nul 2>&1

echo [4/5] Menyimpan snapshot (commit)...
git add -A
git commit -m "Auto-save progres %date% %time%" >nul 2>&1
if errorlevel 1 echo     (Tidak ada perubahan baru, snapshot terakhir tetap aman.)

echo [5/5] Mengunggah ke GitHub...
git remote get-url origin >nul 2>&1
if errorlevel 1 goto BUAT_REPO
for /f "delims=" %%U in ('git remote get-url origin') do set "ORIGIN_URL=%%U"
echo %ORIGIN_URL% | findstr /C:"USERNAME/NAMA_REPO" >nul
if not errorlevel 1 (
  echo     Alamat GitHub lama masih contoh, menggantinya dengan repo asli...
  git remote remove origin
  goto BUAT_REPO
)
git push -u origin main
if errorlevel 1 goto PUSH_GAGAL
goto SELESAI

:BUAT_REPO
where gh >nul 2>&1
if errorlevel 1 goto PASANG_GH
gh auth status >nul 2>&1
if errorlevel 1 (
  echo Silakan login GitHub di browser yang akan terbuka...
  gh auth login -w
)
echo Membuat repo privat "%REPO_NAME%" dan mengunggah...
gh repo create %REPO_NAME% --private --source=. --remote=origin --push
if errorlevel 1 goto PUSH_GAGAL
goto SELESAI

:PASANG_GH
echo.
echo GitHub CLI belum ada. Memasang otomatis (sekali saja)...
winget install --id GitHub.cli -e --source winget
echo.
echo ============================================================
echo   Pemasangan selesai. TUTUP jendela ini,
echo   lalu klik dua kali file ini SEKALI LAGI.
echo ============================================================
pause
exit /b 0

:PUSH_GAGAL
echo.
echo [PERHATIAN] Snapshot lokal AMAN, tapi unggah ke GitHub gagal.
echo Kemungkinan: belum login GitHub, atau koneksi internet bermasalah.
echo Jalankan file ini lagi setelah internet/login beres.
pause
exit /b 1

:SELESAI
echo.
echo ============================================================
echo    SELESAI. Progres tersimpan dan ter-backup ke GitHub.
echo ============================================================
pause
