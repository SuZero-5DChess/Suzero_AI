#!/usr/bin/env python3
"""Run the fixed HC profiling manifest; all search timing occurs in 5dtools."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import re
from pathlib import Path
import signal
import subprocess
import sys


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = Path(__file__).with_name("profile_hc_games.json")
POLICIES = ("balanced", "stable", "deferred", "iterative", "fine", "naive")


def nonnegative(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def positive(value: str) -> int:
    number = nonnegative(value)
    if number == 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--executable", help="override the executable in the manifest")
    parser.add_argument("--policy", choices=POLICIES)
    parser.add_argument("--warmup-rounds", type=nonnegative)
    parser.add_argument("--rounds", type=positive)
    parser.add_argument("--workload", choices=("all", "checkmate", "perftest", "count"), default="all")
    parser.add_argument("--output", type=Path, help="save JSON here (default: logs/hc-profiles/<UTC run id>.json)")
    parser.add_argument("--timeout", type=positive, help="whole-invocation wall-time limit in seconds")
    args = parser.parse_args()

    try:
        manifest = json.loads(args.config.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("manifest must be an object")
        executable = args.executable or manifest.get("executable")
        if not isinstance(executable, str) or not executable:
            raise ValueError("manifest executable must be a nonempty path")
        executable_path = (ROOT / executable).resolve()
        if not executable_path.is_file():
            raise ValueError(f"executable not found: {executable_path}; build 5dtools or use --executable")
        policy = args.policy or manifest.get("policy", "balanced")
        if policy not in POLICIES:
            raise ValueError(f"unknown policy: {policy}")
        warmups = args.warmup_rounds if args.warmup_rounds is not None else manifest.get("warmup_rounds", 3)
        rounds = args.rounds if args.rounds is not None else manifest.get("rounds", 5)
        if type(warmups) is not int or warmups < 0 or type(rounds) is not int or rounds < 1:
            raise ValueError("warmup_rounds must be nonnegative and rounds must be positive integers")
        command = [str(executable_path), "hc-profile"]
        file_count = 0
        inputs = []
        records = []
        for key, option in (("profile-checkmate", "checkmate"), ("profile-perftest", "perftest"),
                            ("profile-count", "count")):
            entries = manifest.get(key, [])
            if not isinstance(entries, list):
                raise ValueError(f"{key} must be an array")
            files = []
            expectations = []
            for entry in entries:
                expected = None
                if option == "count":
                    if not isinstance(entry, dict):
                        raise ValueError("profile-count entries require file and expected fields")
                    filename = entry.get("file")
                    expected = entry.get("expected")
                    if not isinstance(expected, str) or not re.fullmatch(r"[<=>][0-9]+", expected):
                        raise ValueError("expected must be a comparison such as =8, <1000, or >100")
                    value = int(expected[1:])
                    limit = value + (expected[0] != "<")
                    if limit < 1 or limit > 2147483647:
                        raise ValueError("expected comparison requires an action limit in 1..2147483647")
                else:
                    filename = entry
                if not isinstance(filename, str) or not filename:
                    raise ValueError(f"{key} requires nonempty file paths")
                files.append(filename)
                expectations.append(expected)
            if args.workload not in ("all", option) or not files:
                continue
            paths = [(ROOT / path).resolve() for path in files]
            if len(set(paths)) != len(paths):
                raise ValueError(f"duplicate files in {key}")
            for path in paths:
                if not path.is_file():
                    raise ValueError(f"profile input not found: {path}")
            for path, expected in zip(paths, expectations):
                content = path.read_bytes()
                pgn = " ".join(content.decode("utf-8").split())
                prefix = f"EXPECTED{expected} " if expected is not None else ""
                records.append(f"{prefix}{option.upper()} {pgn}\n")
                entry = {"workload": key, "file": str(path),
                         "record": len(records),
                         "sha256": hashlib.sha256(content).hexdigest()}
                if expected is not None:
                    entry["expected"] = expected
                inputs.append(entry)
            file_count += len(paths)
        if file_count == 0:
            raise ValueError("the selected workload has no files")
        command.extend(["--policy", policy, "--warmup-rounds", str(warmups),
                        "--rounds", str(rounds), "--format", "json"])
        executable_hash = hashlib.sha256(executable_path.read_bytes()).hexdigest()
    except (OSError, ValueError) as error:
        parser.error(str(error))

    started = datetime.now(timezone.utc)
    output = args.output or ROOT / "logs" / "hc-profiles" / (started.strftime("%Y%m%dT%H%M%S-%fZ") + ".json")
    metadata = {
        "started_at": started.isoformat(),
        "manifest": manifest,
        "command": command,
        "executable_sha256": executable_hash,
        "inputs": inputs,
    }
    print(f"Profiling {file_count} files with {policy}; output: {output}", file=sys.stderr)
    try:
        # Keep warmups and measured rounds in one native process. stderr carries
        # native progress; stdout is machine-readable profiling data only.
        with subprocess.Popen(
            command, cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, encoding="utf-8",
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        ) as process:
            was_interrupted = False
            try:
                stdout, _ = process.communicate(input="".join(records), timeout=args.timeout)
            except KeyboardInterrupt:
                was_interrupted = True
                # Ctrl+C may already have reached the child through the terminal.
                # Forward it as well for callers that signal only this launcher.
                signal.signal(signal.SIGINT, signal.SIG_IGN)
                print("Interrupted; collecting partial results...", file=sys.stderr)
                if process.poll() is None:
                    try:
                        process.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)
                    except ProcessLookupError:
                        pass
                stdout, _ = process.communicate()
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, _ = process.communicate()
                report = {"error": f"whole profiling invocation exceeded {args.timeout} seconds",
                          "partial_stdout": stdout}
                metadata["status"] = "timeout"
                metadata["exit_code"] = process.returncode
                exit_code = 124
            if metadata.get("status") != "timeout":
                metadata["exit_code"] = process.returncode
                try:
                    report = json.loads(stdout)
                    if not isinstance(report, dict):
                        raise ValueError("expected an object")
                    if process.returncode not in (0, 1, 130) or "summary" not in report:
                        raise ValueError("missing profiling result")
                    was_interrupted |= bool(report.get("interrupted")) or process.returncode == 130
                    metadata["status"] = "interrupted" if was_interrupted else (
                        "ok" if process.returncode == 0 else "error")
                except ValueError:
                    report = {"error": "5dtools did not return a profiling JSON object", "stdout": stdout}
                    metadata["status"] = "interrupted" if was_interrupted else "process_error"
                exit_code = 130 if was_interrupted else (
                    process.returncode if process.returncode > 0 else (0 if metadata["status"] == "ok" else 1))
    except OSError as error:
        report = {"error": str(error)}
        metadata["status"] = "process_error"
        exit_code = 1
    # Native stdin cases use line numbers; retain source filenames in launcher reports.
    filenames = {f"stdin:{entry['record']}": entry["file"] for entry in inputs}
    for case in report.get("cases", []):
        case["file"] = filenames.get(case["file"], case["file"])
    report["run"] = metadata
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except OSError as error:
        print(f"Cannot save profile: {error}", file=sys.stderr)
        return 1
    for case in report.get("cases", []):
        if case["workload"] == "profile-count":
            print(f"profile-count {Path(case['file']).name}: action_count={case.get('action_count')} "
                  f"exhausted={case.get('exhausted')} complete={case['complete']}")
    for workload, summary in report.get("summary", {}).items():
        timing = summary["timing"]
        if timing["samples"] or summary["incomplete_cases"]:
            print(f"{workload}: samples={timing['samples']} mean_ms={timing['mean_ms']} "
                  f"p95_ms={timing['p95_ms']} max_ms={timing['max_ms']} "
                  f"failed_cases={summary['failed_cases']} "
                  f"incomplete_cases={summary['incomplete_cases']}")
    print(f"Saved {output}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
