@echo off
title AI License Plate Recognition System (ALPR) - B.Tech AI/ML Project
echo ================================================================================
echo   AI-BASED AUTOMATIC LICENSE PLATE DETECTION AND RECOGNITION SYSTEM (ALPR)
echo   College B.Tech AI/ML 5th-Semester Capstone Project
echo ================================================================================
echo.
echo [1/3] Setting working directory...
cd /d "%~dp0"

echo [2/3] Checking Python virtual environment...
if not exist "venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment 'venv' not found!
    echo Please create it using: py -3.13 -m venv venv
    pause
    exit /b 1
)

echo [3/3] Launching Streamlit Web Application...
set PYTHONIOENCODING=utf-8
start "" http://localhost:8501
.\venv\Scripts\streamlit.exe run app.py --server.headless=true --server.port=8501
pause
