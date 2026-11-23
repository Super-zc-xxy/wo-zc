@echo off
chcp 65001 >nul
cd /d "%~dp0"

net session >nul 2>&1
if %errorlevel% neq 0 (
    powershell -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
    exit /b
)

set "PYTHON_EXE="
if exist ".\.venv\Scripts\pythonw.exe" (
    set "PYTHON_EXE=.\.venv\Scripts\pythonw.exe"
) else if exist ".\.venv\Scripts\python.exe" (
    set "PYTHON_EXE=.\.venv\Scripts\python.exe"
) else (
    where pythonw >nul 2>&1
    if %errorlevel% equ 0 (
        set "PYTHON_EXE=pythonw.exe"
    ) else (
        where python >nul 2>&1
        if %errorlevel% equ 0 (
            set "PYTHON_EXE=python.exe"
        )
    )
)

if "%PYTHON_EXE%"=="" (
    echo ========================================================
    echo   [错误] 未检测到可用的 Python 运行环境！
    echo   请先安装 Python 3.10+ 并勾选 "Add to PATH"，
    echo   或在当前目录下创建 .venv 虚拟环境：
    echo     python -m venv .venv
    echo     .\.venv\Scripts\pip install -r requirements.txt
    echo ========================================================
    pause
    exit /b 1
)

start "" "%PYTHON_EXE%" main.py
