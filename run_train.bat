@echo off
REM SuZero AI 5D chess AlphaZero training launcher.
REM
REM Requires the pybind11 extension:
REM   python\suzero_5dchess\_5dchess*.pyd
REM
REM Build it once with:
REM   cmake -S . -B build -G "Visual Studio 17 2022" -A x64 ^
REM     -DSUZERO_BUILD_5DCHESS=ON -DSUZERO_BUILD_PYTHON_BINDINGS=ON ^
REM     -DSUZERO_5DCHESS_ENGINE_ROOT=... -DSUZERO_ONNXRUNTIME_ROOT=...
REM
REM Usage: run_train.bat [VARIANT] [ARCHITECTURE] [extra train args...]
REM   VARIANT       very_small (default) | standard_turn_zero (300k budget)
REM                 standard (needs --max-params >= 270202)
REM   ARCHITECTURE  cls_action_mlp (default and only supported value)
REM
REM Training resumes automatically from
REM   runs\<VARIANT>\cls_action_mlp\latest.pt
REM when it exists. Every remaining argument is forwarded to the trainer and
REM overrides the defaults below.
REM
REM Tuning knobs (edit the values near the top of the script):
REM   CONTINUOUS             0 (default) | 1
REM                          Continuous training.  The trainer does not exit
REM                          after --iterations rounds: it saves and resumes
REM                          from latest.pt, then starts the next batch, and
REM                          keeps going until you press Ctrl+C.  --iterations
REM                          is the length of each batch in this mode.  A
REM                          single Ctrl+C finishes the current iteration,
REM                          saves and stops; a second press aborts at once.
REM                          The same can be requested per run with
REM                            run_train.bat very_small cls_action_mlp --continuous
REM   SELFPLAY_BACKEND       cpp_onnx (default) | python
REM                          cpp_onnx exports the live network to ONNX every
REM                          iteration and runs PUCT + ONNX Runtime in C++.
REM                          python keeps the older in-process PyTorch MCTS.
REM   CPP_ONNX_PROVIDER      cuda (default) | cpu
REM                          Device for the C++ self-play backend.  cuda needs
REM                          onnxruntime-gpu (its DLLs are copied next to the
REM                          extension at build time) plus CUDA 12.x / cuDNN 9.
REM                          Use --device cpu|cuda to choose where the PyTorch
REM                          network trains.
REM   SELFPLAY_WORKERS       Total number of concurrent self-play games/threads
REM                          (default 50).  Each game runs its own PUCT
REM                          search; within each process they share that
REM                          process's ONNX evaluator.
REM                          With SELFPLAY_PROCESSES > 1 this is a total budget
REM                          shared by the processes, not a per-process count.
REM   NUM_SERVER_THREADS     Total C++ ONNX evaluator server threads (default 2).
REM                          Only used by the cpp_onnx backend.  Independent of
REM                          SELFPLAY_WORKERS: every server thread owns a
REM                          separate ONNX Session, so raising this costs extra
REM                          RAM and, with CPP_ONNX_PROVIDER=cuda, extra VRAM.
REM                          More threads may improve concurrency, but they do
REM                          not guarantee larger inference batches and are not
REM                          always faster - tune against the actual device.
REM                          With SELFPLAY_PROCESSES > 1 this is a total budget
REM                          shared by the processes.  Override for a single
REM                          run with --num-server-threads N, e.g.
REM                            run_train.bat very_small cls_action_mlp --num-server-threads 4
REM                          An extension built before this option existed
REM                          rejects anything but 1; rebuild it in that case.
REM   SELFPLAY_PROCESSES     Number of self-play worker processes (default 1).
REM                          1 keeps the original single-process training path.
REM                          Values > 1 require SELFPLAY_BACKEND=cpp_onnx and
REM                          SELFPLAY_WORKERS, NUM_SERVER_THREADS and
REM                          SELFPLAY_GAMES each >= SELFPLAY_PROCESSES.
REM                          The worker and server-thread values remain total
REM                          budgets split across processes: they do not have
REM                          to divide evenly (e.g. 50 workers over 2 processes
REM                          can be 25+25; 3 server threads can be 2+1) and are
REM                          never scaled up automatically.  Example:
REM                            run_train.bat very_small cls_action_mlp --selfplay-processes 2
REM   SELFPLAY_GAMES         Self-play games generated per training iteration
REM                          (default 300).  The games are spread over
REM                          SELFPLAY_WORKERS threads, so raising this alone
REM                          makes every iteration roughly
REM                          SELFPLAY_GAMES / SELFPLAY_WORKERS times longer
REM                          without using more of the GPU.  Raise the worker
REM                          count together with it, and keep
REM                          SELFPLAY_GAMES >= SELFPLAY_WORKERS.
REM                          Override for a single run with --games N, e.g.
REM                            run_train.bat very_small cls_action_mlp --games 32
REM   MIN_BOARD_LIMIT / MAX_BOARD_LIMIT
REM                          Board-count range sampled for every self-play game.
REM   EVALUATION_INTERVAL    Iterations between paired C++ model matches;
REM                          0 disables both historical and fixed matches.
REM   MATCH_LOOKBACK         Opponent is exactly this many iterations older.
REM                          Missing history is skipped, never approximated.
REM   MATCH_GAMES / MATCH_SIMS / MATCH_MAX_BOARDS
REM                          Board limit used by periodic model matches.
REM   MATCH_WORKERS          Concurrent C++ games per opponent (default 20).
REM                          Independent of SELFPLAY_WORKERS; 1 is sequential.
REM                          Each model has its own batching evaluator/session.
REM   MATCH_REFERENCE        Additional fixed trusted checkpoint (very_small).
REM                          Other variants use only the historical opponent.
REM   MATCH_PROVIDER         cpu | cuda, independent of self-play/training.
REM                          CLI: --model-match-interval/--model-match-lookback,
REM                          --model-match-games/--model-match-sims, etc.
REM
REM Ctrl+C: the first press finishes the current iteration (self-play,
REM training, evaluation), saves the checkpoint and exits; a second press
REM aborts immediately.  The press is acknowledged on the console at once,
REM including while the C++ self-play batch is running.
REM
REM Examples:
REM   run_train.bat
REM   run_train.bat very_small cls_action_mlp --sims 400 --iterations 5
REM   run_train.bat very_small cls_action_mlp --device cpu
REM   run_train.bat very_small --debug
REM   run_train.bat standard cls_action_mlp --games 16
REM   run_train.bat very_small cls_action_mlp --selfplay-processes 2
REM The last example opts into two self-play processes; omit it (or pass 1) to
REM keep the default single-process path.

setlocal EnableExtensions
set "ROOT=%~dp0"
REM SUZERO_PYTHON can be a full path or an executable name on PATH.
if not defined SUZERO_PYTHON set "SUZERO_PYTHON=python"
set "PYTHON=%SUZERO_PYTHON%"

set "VARIANT=standard_turn_zero"
set "ARCHITECTURE=cls_action_mlp"
if not defined GN5D_DATA_DIR set "GN5D_DATA_DIR=%ROOT%runs\gn5d"

REM Self-play / evaluation knobs.  Extras on the command line are appended
REM after DEFAULT_ARGS and therefore override these values.
set "SELFPLAY_BACKEND=cpp_onnx"
set "CPP_ONNX_PROVIDER=cuda"
set "SELFPLAY_WORKERS=100"
set "NUM_SERVER_THREADS=4"
set "SELFPLAY_PROCESSES=4"
set "SELFPLAY_GAMES=1000"
set "MIN_BOARD_LIMIT=55"
set "MAX_BOARD_LIMIT=90"
set "EVALUATION_INTERVAL=2"
set "MATCH_LOOKBACK=2"
set "MATCH_GAMES=20"
set "MATCH_WORKERS=20"
set "MATCH_SIMS=1000"
set "MATCH_MAX_BOARDS=50"
set "MATCH_PROVIDER=cuda"
REM Optional fixed opponent for very_small matches. The repository does not
REM ship checkpoints; set MATCH_REFERENCE to a local .pt path to enable it.
if not defined MATCH_REFERENCE set "MATCH_REFERENCE="
set "CONTINUOUS=1"

REM Positional argument 1: variant (skip empty or '--flag' arguments).
set "ARG=%~1"
if not defined ARG goto after_positional_1
if not "%ARG:~0,1%"=="-" (
    set "VARIANT=%ARG%"
    shift
)
:after_positional_1

REM Positional argument 2: architecture.
set "ARG=%~1"
if not defined ARG goto after_positional_2
if not "%ARG:~0,1%"=="-" (
    set "ARCHITECTURE=%ARG%"
    shift
)
:after_positional_2

REM Forward every remaining argument verbatim.
set "EXTRA_ARGS="
:collect_args
if "%~1"=="" goto args_ready
set "EXTRA_ARGS=%EXTRA_ARGS% %1"
shift
goto collect_args
:args_ready

if /I not "%ARCHITECTURE%"=="cls_action_mlp" (
    echo [ERROR] Unsupported architecture: %ARCHITECTURE%.
    echo [INFO] This project implements cls_action_mlp only.
    pause
    exit /b 1
)

if /I "%VARIANT%"=="standard" (
    echo [WARN] %VARIANT% exceeds the default 200,000-parameter budget.
    echo [INFO] Pass --max-params 300000 or higher to allow it.
)

REM --debug stores runs\debug for very_small, runs\debug\<VARIANT> otherwise.
set "DEBUG_MODE="
echo %EXTRA_ARGS% | findstr /I /C:"--debug" >nul 2>&1
if not errorlevel 1 set "DEBUG_MODE=1"

REM Continuous training can be enabled by the CONTINUOUS knob or by passing
REM --continuous on the command line (which EXTRA_ARGS forwards anyway).
set "CONTINUOUS_RUN="
if not "%CONTINUOUS%"=="0" set "CONTINUOUS_RUN=1"
echo %EXTRA_ARGS% | findstr /I /C:"--continuous" >nul 2>&1
if not errorlevel 1 set "CONTINUOUS_RUN=1"

set "RUN_DIR=%ROOT%runs\%VARIANT%\cls_action_mlp"
if defined DEBUG_MODE (
    set "RUN_DIR=%ROOT%runs\debug"
    if /I not "%VARIANT%"=="very_small" set "RUN_DIR=%ROOT%runs\debug\%VARIANT%"
)
set "CONSOLE_LOG_DIR=%RUN_DIR%\console"
set "PYTHONPATH=%ROOT%python;%PYTHONPATH%"
cd /d "%ROOT%"

if not exist "%CONSOLE_LOG_DIR%" mkdir "%CONSOLE_LOG_DIR%" >nul 2>&1
if not exist "%CONSOLE_LOG_DIR%" (
    echo [ERROR] Cannot create the console log directory: %CONSOLE_LOG_DIR%
    pause
    exit /b 1
)

REM Accept either a full interpreter path or a bare name resolved via PATH.
set "PYTHON_OK="
if exist "%PYTHON%" set "PYTHON_OK=1"
if not defined PYTHON_OK (
    where "%PYTHON%" >nul 2>&1
    if not errorlevel 1 set "PYTHON_OK=1"
)
if not defined PYTHON_OK (
    echo [ERROR] Python not found: %PYTHON%
    echo [INFO] Set SUZERO_PYTHON to a Python with torch, numpy, onnx and onnxruntime.
    >>"%CONSOLE_LOG_DIR%\launcher_errors.log" echo [%DATE% %TIME%] [ERROR] Python not found: %PYTHON%
    echo [INFO] Error log: %CONSOLE_LOG_DIR%\launcher_errors.log
    pause
    exit /b 1
)

set "EXTENSION_FOUND="
for %%F in ("%ROOT%python\suzero_5dchess\_5dchess*.pyd") do set "EXTENSION_FOUND=%%~fF"
if not defined EXTENSION_FOUND (
    echo [ERROR] The 5D chess Python extension is missing: _5dchess.pyd
    echo [INFO] Reconfigure with -DSUZERO_BUILD_5DCHESS=ON
    echo [INFO] and -DSUZERO_BUILD_PYTHON_BINDINGS=ON, then rebuild.
    >>"%CONSOLE_LOG_DIR%\launcher_errors.log" echo [%DATE% %TIME%] [ERROR] _5dchess extension not found
    echo [INFO] Error log: %CONSOLE_LOG_DIR%\launcher_errors.log
    pause
    exit /b 1
)

if /I "%CPP_ONNX_PROVIDER%"=="cuda" (
    "%PYTHON%" -c "import os, sys, onnxruntime; d = os.path.join(os.path.dirname(onnxruntime.__file__), 'capi'); sys.exit(0 if os.path.isfile(os.path.join(d, 'onnxruntime_providers_cuda.dll')) else 1)" >nul 2>&1
    if errorlevel 1 (
        echo [WARN] onnxruntime_providers_cuda.dll was not found in the
        echo [WARN] installed onnxruntime package. Install onnxruntime-gpu in
        echo [WARN] %PYTHON%, or the CUDA self-play session will fail to start.
    )
)

set "RESUME_ARG="
if exist "%RUN_DIR%\latest.pt" set "RESUME_ARG=--resume latest"

REM Default budget for the C++ self-play backend.  Any flag in EXTRA_ARGS is
REM appended after these and therefore wins.  --debug switches to the built-in
REM tiny preset instead of the budget.
set "EVAL_ARGS=--model-match-interval %EVALUATION_INTERVAL% --model-match-lookback %MATCH_LOOKBACK%"
set "EVAL_ARGS=%EVAL_ARGS% --model-match-games %MATCH_GAMES% --model-match-workers %MATCH_WORKERS% --model-match-sims %MATCH_SIMS% --model-match-max-boards %MATCH_MAX_BOARDS% --model-match-provider %MATCH_PROVIDER%"
if /I "%VARIANT%"=="very_small" if defined MATCH_REFERENCE set EVAL_ARGS=%EVAL_ARGS% --model-match-reference "%MATCH_REFERENCE%"
set "CONTINUOUS_ARG="
if not "%CONTINUOUS%"=="0" set "CONTINUOUS_ARG=--continuous"
set "DEFAULT_ARGS=--iterations 20 --games %SELFPLAY_GAMES% --sims 1000 --save-interval 1 --record-games --gn5d-data-dir "%GN5D_DATA_DIR%""
set "DEFAULT_ARGS=%DEFAULT_ARGS% --selfplay-backend %SELFPLAY_BACKEND% --cpp-onnx-provider %CPP_ONNX_PROVIDER%"
set "DEFAULT_ARGS=%DEFAULT_ARGS% --workers %SELFPLAY_WORKERS%"
set "DEFAULT_ARGS=%DEFAULT_ARGS% --num-server-threads %NUM_SERVER_THREADS%"
set "DEFAULT_ARGS=%DEFAULT_ARGS% --selfplay-processes %SELFPLAY_PROCESSES%"
set "DEFAULT_ARGS=%DEFAULT_ARGS% --min-board-limit %MIN_BOARD_LIMIT% --max-board-limit %MAX_BOARD_LIMIT% %EVAL_ARGS%"
set "DEFAULT_ARGS=%DEFAULT_ARGS% %CONTINUOUS_ARG%"
if defined DEBUG_MODE set "DEFAULT_ARGS=--record-games %CONTINUOUS_ARG%"

echo [INFO] Starting 5D chess AlphaZero training...
echo [INFO] Variant:       %VARIANT%
echo [INFO] Architecture:  %ARCHITECTURE%
echo [INFO] Python:        %PYTHON%
echo [INFO] Run dir:       %RUN_DIR%
if defined RESUME_ARG (
    echo [INFO] Resume:        latest checkpoint
) else (
    echo [INFO] Resume:        fresh run
)
if defined DEBUG_MODE (
    echo [INFO] Self-play:     debug preset - knobs ignored
    echo [INFO] Evaluation:    debug preset - off
) else (
    echo [INFO] Self-play defaults: backend=%SELFPLAY_BACKEND% provider=%CPP_ONNX_PROVIDER% processes=%SELFPLAY_PROCESSES% games/iter=%SELFPLAY_GAMES% workers=%SELFPLAY_WORKERS% server-threads=%NUM_SERVER_THREADS% boards=%MIN_BOARD_LIMIT%-%MAX_BOARD_LIMIT%
    echo [INFO] Model matches: every %EVALUATION_INTERVAL% iterations; lookback=%MATCH_LOOKBACK%; games/opponent=%MATCH_GAMES%; workers=%MATCH_WORKERS%; sims=%MATCH_SIMS%; provider=%MATCH_PROVIDER%; 0 = off
    if /I "%VARIANT%"=="very_small" if defined MATCH_REFERENCE echo [INFO] Fixed opponent: %MATCH_REFERENCE%
)
if defined CONTINUOUS_RUN (
    echo [INFO] Continuous:    on - resumes from latest.pt after each batch until Ctrl+C
) else (
    echo [INFO] Continuous:    off
)
echo [INFO] Extra args:    %EXTRA_ARGS%
echo [INFO] Press Ctrl+C to stop after the current iteration; checkpoints are atomic.
echo [INFO] TensorBoard:   %RUN_DIR%\logs\tensorboard
echo.

"%PYTHON%" -u -m suzero_5dchess train --variant %VARIANT% %DEFAULT_ARGS% %RESUME_ARG% %EXTRA_ARGS%

set "TRAIN_EXIT_CODE=%ERRORLEVEL%"

echo.
if "%TRAIN_EXIT_CODE%"=="0" (
    if defined CONTINUOUS_RUN (
        echo [INFO] Continuous training stopped; progress is saved in latest.pt.
    ) else (
        echo [INFO] Training finished.
    )
) else (
    echo [INFO] Training exited with code %TRAIN_EXIT_CODE%. Unfinished work may not be saved.
)
echo [INFO] Logs and checkpoints: %RUN_DIR%
echo [INFO] Re-run this script to resume from the latest checkpoint.
pause
exit /b %TRAIN_EXIT_CODE%
