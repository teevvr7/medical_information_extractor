@echo off
subst M: "%~dp0" >nul 2>&1
cd /d M:\backend
M:\backend\venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
pause