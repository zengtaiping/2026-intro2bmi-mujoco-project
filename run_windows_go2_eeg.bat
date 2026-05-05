@echo off
setlocal

set PYTHON_EXE=C:\Users\zxzhang\go2-eeg-venv\Scripts\python.exe
set SCRIPT_DIR=%~dp0

"%PYTHON_EXE%" "%SCRIPT_DIR%windows_go2_eeg_keyboard_control.py" --participant test --trials-per-target 20 --trial-duration 5 --iti-duration 3

endlocal
