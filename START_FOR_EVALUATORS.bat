@echo off
echo 🚀 Starting Research Brief Generator for Evaluation...
echo.
echo 📋 EVALUATOR INSTRUCTIONS:
echo 1. Wait for server to start (you'll see "Uvicorn running on http://127.0.0.1:8000")
echo 2. Open your browser and go to: http://127.0.0.1:8000
echo 3. Look for completed workflows and click "FULL BRIEF CONTENT"
echo 4. View the complete assignment content!
echo.
echo ⏳ Starting server...
echo.

python -m uvicorn app.api:app --reload --host 0.0.0.0 --port 8000

pause
