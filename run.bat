@echo off
:: Set code page to UTF-8 for nice icons and text characters
chcp 65001 > nul
setlocal enabledelayedexpansion

title Eyeva AI - Launcher

echo =====================================================================
echo         Eyeva AI V1 Launcher and Environment Setup
echo =====================================================================
echo.

:: 1. Check if python is available
where python >nul 2>nul
if !errorlevel! neq 0 (
    echo [ERROR] Python is not installed or not in your PATH.
    echo Please install Python and try again.
    pause
    exit /b 1
)

:: 2. Check if node/npm is available
where npm >nul 2>nul
if !errorlevel! neq 0 (
    echo [ERROR] Node.js/npm is not installed or not in your PATH.
    echo Please install Node.js and try again.
    pause
    exit /b 1
)

:: 3. Configure environment files if missing
if not exist ".env" (
    echo [INFO] .env file not found. Copying from .env.example...
    copy ".env.example" ".env" >nul
    echo [WARNING] Created .env file. Please edit it and set your NVIDIA_API_KEY!
)

if not exist "frontend\.env.local" (
    echo [INFO] frontend\.env.local not found. Creating it...
    echo NEXT_PUBLIC_API_URL=http://127.0.0.1:8000> "frontend\.env.local"
    echo NEXT_PUBLIC_WS_URL=ws://127.0.0.1:8000>> "frontend\.env.local"
)

:: 4. Locate or create Python virtual environment
set "VENV_DIR=backend\venv"
set "VENV_CREATED=0"

if exist ".venv\Scripts\activate.bat" (
    set "VENV_DIR=.venv"
) else if exist "backend\venv\Scripts\activate.bat" (
    set "VENV_DIR=backend\venv"
) else (
    echo [INFO] Python virtual environment not found. Creating one in backend\venv...
    python -m venv "backend\venv"
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    set "VENV_DIR=backend\venv"
    set "VENV_CREATED=1"
)

echo [SUCCESS] Using virtual environment at: %VENV_DIR%
echo.

:: 5. Install Backend Requirements
if "%VENV_CREATED%"=="1" (
    echo [INFO] Virtual environment was newly created. Installing requirements...
    call :install_reqs
) else (
    set "INSTALL_CHOICE=n"
    set /p "INSTALL_CHOICE=Do you want to check/install Python requirements? (y/n) [default: n]: "
    if /i "!INSTALL_CHOICE!"=="y" (
        call :install_reqs
    )
)

:: 6. Check/Install Frontend Requirements
if not exist "frontend\node_modules" (
    echo [INFO] frontend/node_modules not found. Installing frontend dependencies...
    echo [INFO] Running 'npm install' in frontend...
    cd frontend
    call npm install
    cd ..
) else (
    set "INSTALL_FE_CHOICE=n"
    set /p "INSTALL_FE_CHOICE=Do you want to check/install Frontend npm dependencies? (y/n) [default: n]: "
    if /i "!INSTALL_FE_CHOICE!"=="y" (
        echo [INFO] Running 'npm install' in frontend...
        cd frontend
        call npm install
        cd ..
    )
)

echo.
echo [INFO] Starting Backend and Frontend services...
echo.

:: 7. Launch both services
:: We use start cmd /k so they run concurrently in their own command prompts.
:: This allows the developer to easily see logs and control the processes.
start "Eyeva AI Backend" cmd /k "title Eyeva AI Backend && cd backend && call ..\%VENV_DIR%\Scripts\activate && uvicorn main:app --reload --port 8000"
start "Eyeva AI Frontend" cmd /k "title Eyeva AI Frontend && cd frontend && npm run dev"

echo [SUCCESS] Services started!
echo - Backend: http://127.0.0.1:8000 (API ^& Docs)
echo - Frontend: http://localhost:3000
echo.
echo Press any key to exit this launcher (services will keep running).
pause
exit /b 0

:install_reqs
echo [INFO] Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip
echo [INFO] Installing paddlepaddle (CPU version)...
pip install paddlepaddle
echo [INFO] Installing other requirements from backend\requirements.txt...
pip install -r backend\requirements.txt
echo [SUCCESS] Backend requirements installed.
exit /b 0
