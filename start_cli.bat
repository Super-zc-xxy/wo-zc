@echo off
chcp 65001 >nul
cd /d "%~dp0"

net session >nul 2>&1
if %errorlevel% neq 0 (
    echo 正在申请管理员权限以向游戏窗口发送按键...
    powershell -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
    exit /b
)

echo ==========================================
echo   鸣潮自动任务助手 V2 - 命令行模式
echo   F8: 启动/暂停  |  F7: 接管恢复  |  F9: 退出
echo ==========================================

set "PYTHON_EXE="
if exist ".\.venv\Scripts\python.exe" (
    set "PYTHON_EXE=.\.venv\Scripts\python.exe"
) else (
    where python >nul 2>&1
    if %errorlevel% equ 0 (
        set "PYTHON_EXE=python.exe"
    )
)

if "%PYTHON_EXE%"=="" (
    echo [错误] 未检测到可用的 Python 运行环境，请先安装 Python 3.10+。
    pause
    exit /b 1
)

"%PYTHON_EXE%" main.py --cli
if %errorlevel% neq 0 (
    pause
)
