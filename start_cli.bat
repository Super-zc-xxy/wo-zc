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
".\.venv\Scripts\python.exe" main.py --cli
pause
