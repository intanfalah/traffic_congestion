#!/bin/bash

echo "🚦 Starting Efficient Traffic System"
echo "======================================"

# Kill old processes
pkill -f "python app" 2>/dev/null
sleep 1

# Start server
echo ""
echo "[1/2] Starting server..."
source .venv/bin/activate
python app_efficient.py &
SERVER_PID=$!

# Wait for server
echo "    Waiting for server..."
sleep 5

# Add CCTVs
echo ""
echo "[2/2] Adding CCTVs..."
python add_real_cctvs.py

echo ""
echo "======================================"
echo "✅ System running!"
echo ""
echo "Dashboard: http://127.0.0.1:5005"
echo ""
echo "Press Ctrl+C to stop"
echo "======================================"

# Wait for interrupt
wait $SERVER_PID
