from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence


def _port(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m suzero_5dchess.ui_launcher")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--ui-port", type=_port, default=8080)
    parser.add_argument("--tensorboard-port", type=_port, default=6006)
    parser.add_argument("--tensorboard-logdir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    processes: list[subprocess.Popen[bytes]] = []

    def stop(_signum: int, _frame: object) -> None:
        for process in processes:
            if process.poll() is None:
                process.terminate()

    signal.signal(signal.SIGINT, stop)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, stop)

    processes.append(subprocess.Popen([
        args.python, "-u", "-m", "suzero_5dchess.ui_server",
        "--port", str(args.ui_port),
    ]))

    if args.tensorboard_logdir.is_dir():
        try:
            subprocess.run(
                [args.python, "-c", "import tensorboard"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.CalledProcessError):
            print("[INFO] TensorBoard skipped: not installed in the selected Python.")
        else:
            processes.append(subprocess.Popen([
                args.python, "-u", "-m", "tensorboard.main",
                "--logdir", str(args.tensorboard_logdir),
                "--host", "127.0.0.1",
                "--port", str(args.tensorboard_port),
                "--reload_interval", "5",
            ]))
    else:
        print(f"[INFO] TensorBoard skipped: no log directory at {args.tensorboard_logdir}")

    print("[INFO] Press Ctrl+C to stop the UI and TensorBoard.")
    try:
        while True:
            for process in processes:
                if process.poll() is not None:
                    return process.returncode or 0
            time.sleep(0.2)
    except (KeyboardInterrupt, SystemExit):
        stop(signal.SIGINT, None)
    finally:
        for process in processes:
            if process.poll() is None:
                process.wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
