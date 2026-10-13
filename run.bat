@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Chua cai thu vien. Chay: powershell -NoProfile -ExecutionPolicy Bypass -File install.ps1
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -X utf8 main.py %*
if errorlevel 1 pause
