@echo off
title Visual Assistant - Complete Setup
color 0A

echo ================================================================
echo           VISUAL ASSISTANT SETUP FOR WINDOWS
echo   AI-Powered Assistance for Visually Impaired Users
echo ================================================================
echo.

REM Check Python installation
echo [1/4] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo ERROR: Python is not installed!
    echo Please install Python 3.8 or higher from:
    echo https://www.python.org/downloads/
    echo.
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

python --version
echo Python found! ✓
echo.

REM Upgrade pip
echo [2/4] Upgrading pip...
python -m pip install --upgrade pip --quiet
echo pip upgraded! ✓
echo.

REM Install dependencies
echo [3/4] Installing required packages...
echo This may take a few minutes...
echo.

pip install requests opencv-python pygame Pillow --quiet

if errorlevel 1 (
    echo.
    echo ERROR: Failed to install dependencies!
    pause
    exit /b 1
)

echo All packages installed! ✓
echo.

REM Verify installation
echo [4/4] Verifying installation...
python -c "import requests, cv2, pygame" 2>nul
if errorlevel 1 (
    echo WARNING: Some packages may not be installed correctly.
    echo You can still try running the application.
    echo.
) else (
    echo Installation verified! ✓
    echo.
)

echo ================================================================
echo                    SETUP COMPLETE!
echo ================================================================
echo.
echo To run Visual Assistant:
echo   1. Double-click run.bat
echo   OR
echo   2. Run: python gui.py
echo.
echo Keyboard Shortcuts:
echo   Alt+L - Load Image
echo   Alt+A - Analyze
echo   F5    - Analyze
echo.
echo For more information, see README.md
echo.
echo ================================================================
pause

REM Ask if user wants to launch now
set /p LAUNCH="Do you want to launch Visual Assistant now? (Y/N): "
if /i "%LAUNCH%"=="Y" (
    python gui.py
)
