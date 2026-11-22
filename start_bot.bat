@echo off
chcp 65001 >nul
cd /d "%~dp0"
if exist "MingchaoJB.exe" (
    start "" "MingchaoJB.exe"
    exit /b
)
net session >nul 2>&1
if %errorlevel% neq 0 (
    powershell -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
    exit /b
)
".\.venv\Scripts\pythonw.exe" main.py
