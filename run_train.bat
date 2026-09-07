@echo off
REM AlphaZero Very Small training launcher (GPU) - continuous training mode
REM Requires: build_py_ml/engine.cp311-win_amd64.pyd

set ROOT=%~dp0
set PYTHON=C:\UserData\Program\anaconda3\envs\dl311\python.exe
cd /d "%ROOT%"

if not exist "%ROOT%build_py_ml\engine.cp311-win_amd64.pyd" (
    echo [ERROR] engine.pyd not found in build_py_ml/
    exit /b 1
)

REM Start the UI server (with API endpoint for live data)
start /B "" "%PYTHON%" serve_ui.py 8080

echo [INFO] Starting AlphaZero training (GPU: RTX 5060)...
echo [INFO] Variant: very_small   Games: 50/iter   Sims: 100   Continuous mode
echo [INFO] Live dashboard: http://127.0.0.1:8080/ui/training.html
echo [INFO] Press Ctrl+C to stop training gracefully (data will be saved)
echo.

"%PYTHON%" -m alphazero.train --variant very_small --selfplay-backend cpp_onnx --games 50 --sims 200 --iterations 20 --continuous --min-board-limit 40 --max-board-limit 80 --pgn-snapshot-interval 49

echo.
echo [INFO] Training finished (or interrupted). Checkpoint saved to alphazero/checkpoints/very_small/
echo [INFO] Log file: alphazero/logs/very_small/training_log.jsonl
echo [INFO] Live dashboard: http://127.0.0.1:8080/ui/training.html
echo.
echo [INFO] Re-run this script to resume training from the latest checkpoint.
pause
