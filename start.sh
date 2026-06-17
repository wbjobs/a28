#!/bin/bash

set -e

echo "========================================"
echo "  Video Understanding App - Launcher"
echo "========================================"
echo ""

if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 not found. Please install Python 3.10+."
    exit 1
fi

if ! command -v node &> /dev/null; then
    echo "[ERROR] Node.js not found. Please install Node.js 18+."
    exit 1
fi

if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "[INFO] Created .env from .env.example"
    fi
fi

echo "[1/4] Checking Python dependencies..."
if [ ! -d "python/venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv python/venv
fi

source python/venv/bin/activate
pip install -r python/requirements.txt -q
echo "Python dependencies ready."

echo ""
echo "[2/4] Checking Node.js dependencies..."
if [ ! -d "node_modules" ]; then
    echo "Installing Node.js dependencies..."
    npm install
fi
echo "Node.js dependencies ready."

echo ""
echo "[3/4] Starting Python backend on port 5000..."
cd python && python app.py &
PYTHON_PID=$!
cd ..

sleep 3

echo ""
echo "[4/4] Starting Node.js server on port 3000..."
echo ""
echo "========================================"
echo "  Services Started!"
echo "  - Python API: http://localhost:5000"
echo "  - Node API:   http://localhost:3000"
echo "  - Frontend:   http://localhost:3000"
echo "========================================"
echo ""
echo "Press Ctrl+C to stop all services."
echo ""

cleanup() {
    echo ""
    echo "Stopping services..."
    kill $PYTHON_PID 2>/dev/null || true
    exit 0
}

trap cleanup SIGINT SIGTERM

npm start
