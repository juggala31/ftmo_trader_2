@echo off
REM Force UTF-8 for Python + console
chcp 65001 >NUL
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

REM Optional: nicer Python logs
set PYTHONUNBUFFERED=1
set PYTHONFAULTHANDLER=1

REM Run
python "%~dp0run_live.py"
pause
