@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo 常用模板名示例:
echo   btn_skip.png / btn_confirm.png / dialogue_option.png
echo   icon_auto_play.png / icon_f.png / icon_t.png
echo   quest_marker.png / in_combat.png
echo.
set /p TPL_NAME=请输入要保存的模板文件名: 
if "%TPL_NAME%"=="" (
    echo 未输入文件名，已取消。
    pause
    exit /b
)
".\.venv\Scripts\python.exe" -m tools.capture_template --name "%TPL_NAME%"
pause
