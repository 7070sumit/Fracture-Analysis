#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "==========================================="
echo " Starting Bone Fracture Detection App"
echo "==========================================="

ROOT_DIR="$(pwd)"

# 1. Setup Virtual Environment outside both directories
echo "1. Checking/Setting up Python Virtual Environment..."
if [ ! -d "venv" ]; then
    echo "Creating virtual environment 'venv'..."
    python3 -m venv venv
else
    echo "Virtual environment 'venv' already exists."
fi

# Activate the virtual environment
source venv/bin/activate
echo "Virtual environment activated."

# 2. Install Backend Dependencies
echo "2. Installing backend dependencies..."
pip install --upgrade pip
pip install -r "$ROOT_DIR/backend/requirements.txt"

# 3. Install Frontend Dependencies
echo "3. Installing frontend dependencies..."
cd "$ROOT_DIR/frontend"
npm install --legacy-peer-deps
cd "$ROOT_DIR"

# 4. Run both concurrently
echo "4. Starting both servers..."

# Start the Backend Server in the background
echo "Starting backend (FastAPI)..."
cd "$ROOT_DIR/backend"
# Run uvicorn in the background
uvicorn server:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
cd "$ROOT_DIR"

# Start the Frontend Server in the background
echo "Starting frontend (React)..."
cd "$ROOT_DIR/frontend"
# Run npm start in the background
npm start &
FRONTEND_PID=$!
cd "$ROOT_DIR"

echo "==========================================="
echo " Both servers are running!"
echo " Backend API is starting on http://localhost:8000"
echo " Frontend App is starting on http://localhost:3000"
echo " Backend PID: $BACKEND_PID"
echo " Frontend PID: $FRONTEND_PID"
echo " Press Ctrl+C to stop both servers."
echo "==========================================="

# Trap Ctrl+C (SIGINT) to kill both background processes
trap "echo -e '\nShutting down both servers...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" SIGINT SIGTERM

# Wait for background processes
wait $BACKEND_PID $FRONTEND_PID
