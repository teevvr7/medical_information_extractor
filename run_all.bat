@echo off
subst M: "%~dp0" >nul 2>&1
start "Medical Info Extractor - Backend" cmd /c "cd /d M:\backend && M:\backend\venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000"
timeout /t 3 >nul
start "Medical Info Extractor - Frontend" cmd /c "cd /d M:\frontend && M:\backend\venv\Scripts\python.exe -m streamlit run app.py --server.port 8501"
timeout /t 2 >nul
start http://localhost:8501