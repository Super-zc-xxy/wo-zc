@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo 正在启动 ROI 与 OCR 实时调试窗口（按 q 退出）...
".\.venv\Scripts\python.exe" -m tools.debug_overlay
pause
