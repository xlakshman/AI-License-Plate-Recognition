@echo off
title AI-Based Automatic License Plate Recognition System (ANPR)
color 0B

echo ======================================================================
echo   AI-Based Automatic License Plate Recognition System (ANPR)
echo   B.Tech CSE (AI ^& ML) College Project ^& Viva Examination
echo ======================================================================
echo.

:: Detect Python Environment
if exist "venv_py311\Scripts\python.exe" (
    echo [*] Utilizing configured Python 3.11 Virtual Environment...
    set PYTHON_EXE=venv_py311\Scripts\python.exe
) else if exist "venv\Scripts\python.exe" (
    echo [*] Utilizing Virtual Environment...
    set PYTHON_EXE=venv\Scripts\python.exe
) else (
    echo [*] Virtual environment not detected, searching system Python...
    where python >nul 2>nul
    if %errorlevel% equ 0 (
        set PYTHON_EXE=python
    ) else (
        echo [ERROR] Python not found in PATH or virtual environments!
        pause
        exit /b 1
    )
)

echo [*] Starting Flask ANPR Web Server...
echo [*] Open your web browser at: http://127.0.0.1:5000
echo.

%PYTHON_EXE% app.py

pause
