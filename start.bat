@echo off
setlocal enabledelayedexpansion

echo ========================================
echo   Video Understanding App - Launcher
echo ========================================
echo.

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.10+.
    exit /b 1
)

where node >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Node.js not found. Please install Node.js 18+.
    exit /b 1
)

if not exist ".env" (
    if exist ".env.example" (
        copy .env.example .env >nul
        echo [INFO] Created .env from .env.example
    )
)

echo [1/4] Checking Python dependencies...
if not exist "python\venv" (
    echo Creating Python virtual environment...
    python -m venv python\venv
)

call python\venv\Scripts\activate.bat
pip install -r python\requirements.txt -q
echo Python dependencies ready.

echo.
echo [2/4] Checking Node.js dependencies...
if not exist "node_modules" (
    echo Installing Node.js dependencies...
    call npm install
)
echo Node.js dependencies ready.

echo.
echo [3/4] Starting Python backend on port 5000...
start "Python Backend" cmd /k "cd /d %~dp0python && ..\python\venv\Scripts\python.exe app.py"

timeout /t 3 /nobreak >nul

echo.
echo [4/4] Starting Node.js server on port 3000...
echo.
echo ========================================
echo   Services Started!
echo   - Python API: http://localhost:5000
echo   - Node API:   http://localhost:3000
echo   - Frontend:   http://localhost:3000
echo ========================================
echo.
echo Press Ctrl+C to stop the Node.js server.
echo.

call npm start
