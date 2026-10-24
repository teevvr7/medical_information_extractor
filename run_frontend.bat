@echo off
subst M: "%~dp0" >nul 2>&1
cd /d M:\frontend
M:\backend\venv\Scripts\python.exe -m streamlit run app.py --server.port 8501
pause