@echo off
title Visual Assistant for Visually Impaired
echo ================================================
echo   Visual Assistant - Starting...
echo ================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://python.org
    pause
    exit /b 1
)

echo Python found!
echo.

REM Check if dependencies are installed
echo Checking dependencies...
python -c "import requests" >nul 2>&1
if errorlevel 1 (
    echo Installing required packages...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
) else (
    echo All dependencies installed!
)

echo.
echo ================================================
echo   Launching Visual Assistant GUI...
echo ================================================
echo.
echo Keyboard Shortcuts:
echo   Alt+L - Load Image
echo   Alt+A or F5 - Analyze
echo.

REM Start the GUI
python gui.py

if errorlevel 1 (
    echo.
    echo ERROR: Application encountered an error
    pause
)
