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
REM   CONTINUOUS             1 (default) | 0
REM                          Continuous training.  The trainer does not exit
REM                          after --iterations rounds: it saves and resumes
REM                          from latest.pt, then starts the next batch, and
REM                          keeps going until you press Ctrl+C.  --iterations
REM                          is the length of each batch in this mode.  A
REM                          single Ctrl+C during self-play stops handing out
REM                          new games, lets the in-flight games finish and
REM                          saves all their samples (skipping training); the
REM                          next run resumes from that progress.  During
REM                          training/evaluation it finishes the iteration,
REM                          saves and stops.  A second press aborts at once.
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
REM   TRAIN_EPOCHS          Training epochs per iteration (default 15).
REM                          Rows trained per iteration =
REM                          TRAIN_EPOCHS x steps-per-epoch(10) x batch(256),
REM                          i.e. how much of the replay window one iteration
REM                          covers.  At 15 that is 38400 rows per iteration;
REM                          the 100000-row window still needs self-play churn
REM                          and several iterations to be covered fully.
REM   REPLAY_BACKEND        file (default) | memory
REM                          file appends each finished self-play game to one
REM                          row log (replay_store\replay.log) the moment the
REM                          game ends and trains from it, so the window is
REM                          never held in RAM.  The window is trimmed by
REM                          position row, in batches, with room to spare.
REM                          memory keeps the whole round in an in-process
REM                          ReplayBuffer and only writes it at the end of the
REM                          iteration.
REM   NO_REPEAT_ROWS        0 (default) | 1; with 1 each shuffle round draws
REM                          only rows the previous round actually trained, so
REM                          rows it selected but never served stay eligible.
REM                          Rows proven trained are disjoint across adjacent
REM                          rounds.  0 samples every round from the whole
REM                          pool instead; rounds may then overlap, which
REM                          halves the sample budget a small pool can offer
REM                          across two rounds.
REM   WINDOW_MODE            0 (default) | 1; opt into dynamic replay growth.
REM                          WINDOW_EXPAND_PER_ROW / WINDOW_TAPER_EXPONENT set
REM                          the growth curve; WINDOW_MAX_SAMPLES is its cap.
REM                          The cap is soft: rows the trainer has not served
REM                          yet are never evicted, so the window can stay
REM                          above it until training reaches them.
REM                          Optional WINDOW_TAPER_SCALE defaults to replay size;
REM                          WINDOW_KEEP_SAMPLES limits only the sampled pool.
REM                          With REPLAY_BACKEND=file each iteration shuffles
REM                          the file list and visits every row once without a
REM                          RAM copy; with memory, shuffle publishes NPZ shards
REM                          once per iteration.  Neither repeats rows within a
REM                          generation; exhaustion returns to self-play.
REM                          CLI --window and --window-* override these knobs.
REM   ENDGAME_SAMPLING      1 (default) | 0; repeat replay rows that are close
REM                          to the end of their game.  Every eligible row is
REM                          still served once per shuffle round; rows nearer
REM                          the end get extra copies, so they are trained on
REM                          more often without dropping any other sample.
REM                          ENDGAME_WEIGHT is the expected extra copies of a
REM                          game's final position (default 1.0); ENDGAME_EXPONENT
REM                          shapes the ramp (default 2.0, concentrated in the
REM                          last part of the game; 1 is linear).
REM   TREE_POSITIONS         1 (default) | 0; reuse full-search side nodes.
REM                          No extra searches; tree values are search targets,
REM                          not the result of the actual played game.
REM                          TREE_MIN_VISITS / TREE_MAX_DEPTH / TREE_MAX_PER_TURN
REM                          bound quality, small-step depth and rows per search.
REM                          TREE_SAMPLE_PROB controls fractional row retention.
REM                          TREE_POSITIONS=0 appends --no-record-tree-positions.
REM                          Default-on cpp_onnx requires a rebuilt extension.
REM   GATING                 1 (default) | 0; candidate promotion matches.
REM                          Set GATING=0 to self-play with the training net as before.
REM                          GATE_INTERVAL / GATE_GAMES / GATE_WORKERS / GATE_SIMS
REM                          set the match schedule and search budget.  The
REM                          interval is in completed training iterations
REM                          (3 = promote-check every third iteration).
REM                          GATE_MIN_AVERAGE_SCORE / GATE_MIN_FINISHED set
REM                          promotion criteria (mean game score, forced
REM                          material terminals included).
REM                          GATE_PROVIDER / GATE_MAX_BOARDS select device and limit.
REM                          External environment values and CLI flags are respected.
REM   MIN_BOARD_LIMIT / MAX_BOARD_LIMIT
REM                          Board-count range sampled for every self-play game.
REM   SUZERO_DISABLE_QUICK_EDIT
REM                          1 turns off the console's QuickEdit for this run.
REM                          Off by default because QuickEdit is what lets the
REM                          mouse select and copy text.  Training no longer
REM                          stalls on a paused console either way: console
REM                          output is written from a background thread and
REM                          dropped if the console is paused, while
REM                          training.log keeps every record.
REM
REM Periodic model matches are not part of this launcher (they are not the
REM gate).  They stay available through the trainer CLI, e.g.
REM   run_train.bat very_small cls_action_mlp --model-match-interval 10 ^
REM     --model-match-lookback 30 --model-match-games 8
REM See README section 3.5 for the full set of --model-match-* flags.
REM
REM Ctrl+C: during self-play the first press stops handing out new games,
REM lets the in-flight games finish, saves all their samples to the replay
REM store (training is skipped) and exits; the next run resumes from that
REM progress.  During training/evaluation the first press finishes the
REM current iteration, saves the checkpoint and exits; a second press
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

REM Self-play knobs.  Extras on the command line are appended
REM after DEFAULT_ARGS and therefore override these values.
set "SELFPLAY_BACKEND=cpp_onnx"
set "CPP_ONNX_PROVIDER=cuda"
set "SELFPLAY_WORKERS=100"
set "NUM_SERVER_THREADS=4"
set "SELFPLAY_PROCESSES=4"
set "SELFPLAY_GAMES=1000"
set "MIN_BOARD_LIMIT=60"
set "MAX_BOARD_LIMIT=90"
set "GATING=0"
set "CONTINUOUS=1"
REM Training budget per iteration: epochs x steps-per-epoch(10) x batch(256).
if not defined TRAIN_EPOCHS set "TRAIN_EPOCHS=15"
REM Per-game file replay is the default; switch to memory to restore the
REM old "keep the whole round in RAM" behaviour.
set "REPLAY_BACKEND=file"
REM 0 samples every round from the whole replay pool, so consecutive rounds may
REM repeat rows; 1 keeps adjacent shuffle rounds disjoint.
if not defined NO_REPEAT_ROWS set "NO_REPEAT_ROWS=0"

REM Dynamic replay window is opt-in; mode 0 leaves the training command unchanged.
REM These affect replay only, never the self-play/search/training budget.
set "WINDOW_MODE=0"
set "WINDOW_EXPAND_PER_ROW=0.4"
set "WINDOW_TAPER_EXPONENT=0.65"
REM Empty scale uses --replay-size; empty keep samples the whole window.
set "WINDOW_TAPER_SCALE="
set "WINDOW_MAX_SAMPLES=500000"
set "WINDOW_KEEP_SAMPLES="
set "WINDOW_SAMPLING=shuffle"

REM Endgame oversampling is on by default; external environment values are respected.
if not defined ENDGAME_SAMPLING set "ENDGAME_SAMPLING=1"
if not defined ENDGAME_WEIGHT set "ENDGAME_WEIGHT=1.0"
if not defined ENDGAME_EXPONENT set "ENDGAME_EXPONENT=2.0"

REM Tree-position reuse is on by default; external environment values are respected.
if not defined TREE_POSITIONS set "TREE_POSITIONS=1"
if not defined TREE_MIN_VISITS set "TREE_MIN_VISITS=64"
if not defined TREE_MAX_DEPTH set "TREE_MAX_DEPTH=5"
if not defined TREE_MAX_PER_TURN set "TREE_MAX_PER_TURN=4"
if not defined TREE_SAMPLE_PROB set "TREE_SAMPLE_PROB=0.25"

REM Candidate promotion is on by default, including in debug mode; GATING=0 disables.
if not defined GATING set "GATING=1"
if not defined GATE_INTERVAL set "GATE_INTERVAL=3"
if not defined GATE_GAMES set "GATE_GAMES=100"
if not defined GATE_WORKERS set "GATE_WORKERS=20"
if not defined GATE_SIMS set "GATE_SIMS=200"
if not defined GATE_MIN_AVERAGE_SCORE set "GATE_MIN_AVERAGE_SCORE=0.05"
if not defined GATE_MIN_FINISHED set "GATE_MIN_FINISHED=50"
if not defined GATE_PROVIDER set "GATE_PROVIDER=cuda"
if not defined GATE_MAX_BOARDS set "GATE_MAX_BOARDS=50"

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
set "CONTINUOUS_ARG="
if not "%CONTINUOUS%"=="0" set "CONTINUOUS_ARG=--continuous"
set "DEFAULT_ARGS=--iterations 20 --games %SELFPLAY_GAMES% --sims 1000 --save-interval 1 --record-games --gn5d-data-dir "%GN5D_DATA_DIR%""
set "DEFAULT_ARGS=%DEFAULT_ARGS% --selfplay-backend %SELFPLAY_BACKEND% --cpp-onnx-provider %CPP_ONNX_PROVIDER%"
set "DEFAULT_ARGS=%DEFAULT_ARGS% --workers %SELFPLAY_WORKERS%"
set "DEFAULT_ARGS=%DEFAULT_ARGS% --num-server-threads %NUM_SERVER_THREADS%"
set "DEFAULT_ARGS=%DEFAULT_ARGS% --selfplay-processes %SELFPLAY_PROCESSES%"
set "DEFAULT_ARGS=%DEFAULT_ARGS% --replay-backend %REPLAY_BACKEND%"
set "DEFAULT_ARGS=%DEFAULT_ARGS% --epochs-per-iteration %TRAIN_EPOCHS%"
if "%NO_REPEAT_ROWS%"=="0" set "DEFAULT_ARGS=%DEFAULT_ARGS% --repeat-rows"
if not "%NO_REPEAT_ROWS%"=="0" set "DEFAULT_ARGS=%DEFAULT_ARGS% --no-repeat-rows"
set "DEFAULT_ARGS=%DEFAULT_ARGS% --min-board-limit %MIN_BOARD_LIMIT% --max-board-limit %MAX_BOARD_LIMIT%"
set "DEFAULT_ARGS=%DEFAULT_ARGS% %CONTINUOUS_ARG%"
if defined DEBUG_MODE set "DEFAULT_ARGS=--record-games %CONTINUOUS_ARG%"

REM Append only when enabled so mode 0 preserves the original command exactly.
if "%WINDOW_MODE%"=="1" set "DEFAULT_ARGS=%DEFAULT_ARGS% --window"
if "%WINDOW_MODE%"=="1" if defined WINDOW_EXPAND_PER_ROW set "DEFAULT_ARGS=%DEFAULT_ARGS% --window-expand-per-row %WINDOW_EXPAND_PER_ROW%"
if "%WINDOW_MODE%"=="1" if defined WINDOW_TAPER_EXPONENT set "DEFAULT_ARGS=%DEFAULT_ARGS% --window-taper-exponent %WINDOW_TAPER_EXPONENT%"
if "%WINDOW_MODE%"=="1" if defined WINDOW_TAPER_SCALE set "DEFAULT_ARGS=%DEFAULT_ARGS% --window-taper-scale %WINDOW_TAPER_SCALE%"
if "%WINDOW_MODE%"=="1" if defined WINDOW_MAX_SAMPLES set "DEFAULT_ARGS=%DEFAULT_ARGS% --window-max-samples %WINDOW_MAX_SAMPLES%"
if "%WINDOW_MODE%"=="1" if defined WINDOW_KEEP_SAMPLES set "DEFAULT_ARGS=%DEFAULT_ARGS% --window-keep-samples %WINDOW_KEEP_SAMPLES%"
if "%WINDOW_MODE%"=="1" if defined WINDOW_SAMPLING set "DEFAULT_ARGS=%DEFAULT_ARGS% --window-sampling %WINDOW_SAMPLING%"

REM Endgame oversampling stays opt-in in the library, so the launcher passes it
REM explicitly (and --debug keeps the same behaviour).
if "%ENDGAME_SAMPLING%"=="1" set "DEFAULT_ARGS=%DEFAULT_ARGS% --endgame-sampling --endgame-weight %ENDGAME_WEIGHT% --endgame-exponent %ENDGAME_EXPONENT%"
if not "%ENDGAME_SAMPLING%"=="1" set "DEFAULT_ARGS=%DEFAULT_ARGS% --no-endgame-sampling"

REM Tree knobs are on by default; TREE_POSITIONS=0 disables the feature even
REM with --debug, so the library default cannot silently re-enable it.
if "%TREE_POSITIONS%"=="1" set "DEFAULT_ARGS=%DEFAULT_ARGS% --record-tree-positions --tree-min-visits %TREE_MIN_VISITS% --tree-max-depth %TREE_MAX_DEPTH% --tree-max-per-turn %TREE_MAX_PER_TURN% --tree-sample-prob %TREE_SAMPLE_PROB%"
if "%TREE_POSITIONS%"=="0" set "DEFAULT_ARGS=%DEFAULT_ARGS% --no-record-tree-positions"

REM Candidate promotion is enabled in the library default, so GATING=0 must pass
REM --gate-interval 0 explicitly; omitting the flag would silently re-enable it.
if "%GATING%"=="1" set "DEFAULT_ARGS=%DEFAULT_ARGS% --gate-interval %GATE_INTERVAL% --gate-games %GATE_GAMES% --gate-workers %GATE_WORKERS% --gate-simulations %GATE_SIMS% --gate-min-average-score %GATE_MIN_AVERAGE_SCORE% --gate-min-finished %GATE_MIN_FINISHED% --gate-provider %GATE_PROVIDER% --gate-max-boards %GATE_MAX_BOARDS%"
if "%GATING%"=="0" set "DEFAULT_ARGS=%DEFAULT_ARGS% --gate-interval 0"

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
) else (
    echo [INFO] Self-play defaults: backend=%SELFPLAY_BACKEND% provider=%CPP_ONNX_PROVIDER% processes=%SELFPLAY_PROCESSES% games/iter=%SELFPLAY_GAMES% workers=%SELFPLAY_WORKERS% server-threads=%NUM_SERVER_THREADS% boards=%MIN_BOARD_LIMIT%-%MAX_BOARD_LIMIT%
    echo [INFO] Replay:        backend=%REPLAY_BACKEND%; file = one row log, replay.log; no-repeat-rows=%NO_REPEAT_ROWS%
)
echo [INFO] Model matches: off by default - pass --model-match-interval N to enable
if "%GATING%"=="1" (
    echo [INFO] Gating:        on; interval=%GATE_INTERVAL%; games=%GATE_GAMES%; workers=%GATE_WORKERS%; sims=%GATE_SIMS%; min-average-score=%GATE_MIN_AVERAGE_SCORE%; min-finished=%GATE_MIN_FINISHED%; provider=%GATE_PROVIDER%; max-boards=%GATE_MAX_BOARDS%
) else (
    echo [INFO] Gating:        off
)
if defined CONTINUOUS_RUN (
    echo [INFO] Continuous:    on - resumes from latest.pt after each batch until Ctrl+C
) else (
    echo [INFO] Continuous:    off
)
echo [INFO] Extra args:    %EXTRA_ARGS%
echo [INFO] Press Ctrl+C to stop; in-flight self-play games finish and their samples are saved.
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
