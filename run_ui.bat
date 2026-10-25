@echo off
setlocal
REM Launch the bundled engine UI with native GN5D replay, plus optional TensorBoard.
REM Override PYTHON, UI_PORT, TENSORBOARD_PORT, or TENSORBOARD_LOGDIR as needed.

set "ROOT=%~dp0"
REM PYTHON can be a full path or an executable name on PATH.
if not defined PYTHON set "PYTHON=python"
if not defined UI_PORT set "UI_PORT=8080"
if not defined TENSORBOARD_PORT set "TENSORBOARD_PORT=6006"
if not defined TENSORBOARD_LOGDIR set "TENSORBOARD_LOGDIR=%ROOT%runs"
set "PYTHONPATH=%ROOT%python;%PYTHONPATH%"

cd /d "%ROOT%"

set "PYTHON_OK="
if exist "%PYTHON%" set "PYTHON_OK=1"
if not defined PYTHON_OK (
    where "%PYTHON%" >nul 2>&1
    if not errorlevel 1 set "PYTHON_OK=1"
)
if not defined PYTHON_OK (
    echo [ERROR] Python not found: %PYTHON%
    echo [INFO] Set PYTHON to a Python executable path or name.
    pause
    exit /b 1
)

if not exist "%ROOT%5dchess_engine-main\ui\scripts\gn5d-worker.js" (
    echo [ERROR] Engine UI or GN5D adapter is missing from 5dchess_engine-main.
    pause
    exit /b 1
)

"%PYTHON%" -c "from suzero_5dchess.env import GameEnv; GameEnv().record_snapshot()"
if errorlevel 1 (
    echo [ERROR] The native 5D chess extension is unavailable or outdated.
    echo [INFO] Build it: cmake --build build --config Release --target suzero_5dchess_py
    pause
    exit /b 1
)

echo [INFO] Engine UI: http://127.0.0.1:%UI_PORT%/
echo [INFO] Use Import - Open File - Load to replay a .gn5d or .5dck record.
echo [INFO] UI assets: %ROOT%5dchess_engine-main\ui (native backend, no WASM build needed).
echo [INFO] TensorBoard: http://127.0.0.1:%TENSORBOARD_PORT%/ (when available)
echo [INFO] Press Ctrl+C to stop the UI and TensorBoard.
"%PYTHON%" -u -m suzero_5dchess.ui_launcher --python "%PYTHON%" --ui-port %UI_PORT% --tensorboard-port %TENSORBOARD_PORT% --tensorboard-logdir "%TENSORBOARD_LOGDIR%"
exit /b %errorlevel%
