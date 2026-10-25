from __future__ import annotations

import argparse
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
from typing import Sequence
from urllib.parse import unquote, urlsplit

from . import env as env_module
from .game_record import FORMAT, VERSION, replay_record


_MAX_REQUEST_BYTES = 16 * 1024 * 1024
_MAX_FRAMES_BYTES = 128 * 1024 * 1024
_ALLOWED_HOSTS = frozenset({"127.0.0.1", "localhost"})
_STATIC_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
    ".ttf": "font/ttf",
    ".wasm": "application/wasm",
}
_ASSET_DIRECTORIES = frozenset({"scripts", "styles", "assets", "fonts", "icons", "wasm"})
_LOGGER = logging.getLogger(__name__)


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant {value!r}")


def _parse_host(value: str, expected_port: int) -> str:
    if value != value.strip() or value.count(":") != 1:
        raise ValueError("invalid Host header")
    hostname, raw_port = value.rsplit(":", 1)
    hostname = hostname.lower()
    if hostname not in _ALLOWED_HOSTS or not raw_port.isascii() or not raw_port.isdigit():
        raise ValueError("invalid Host header")
    port = int(raw_port)
    if port != expected_port:
        raise ValueError("Host port does not match the server port")
    return hostname


def _validate_origin(value: str, hostname: str, expected_port: int) -> None:
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise ValueError("invalid Origin header") from exc
    if (
        parsed.scheme.lower() != "http"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.hostname is None
        or parsed.hostname.lower() != hostname
        or parsed.hostname.lower() not in _ALLOWED_HOSTS
        or (80 if port is None else port) != expected_port
        or parsed.path
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("cross-origin requests are not allowed")


def _project_ui_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "5dchess_engine-main" / "ui"


def _check_ui_dir(ui_dir: Path) -> None:
    required = (
        "index.html", "scripts/main.js", "scripts/draw.js", "scripts/ui.js",
        "scripts/gn5d-worker.js", "scripts/gn5d-replay.js", "styles/main.css",
        "assets/king-white.svg",
    )
    missing = [name for name in required if not (ui_dir / name).is_file()]
    if missing:
        raise RuntimeError(
            f"Engine UI assets were not found in {ui_dir}: {', '.join(missing)}. "
            "Keep 5dchess_engine-main beside this project's CMakeLists.txt."
        )


def _static_asset(ui_dir: Path, request_path: str) -> Path | None:
    parsed = urlsplit(request_path)
    if parsed.query or parsed.fragment:
        return None
    path = unquote(parsed.path)
    if "\\" in path or "\x00" in path:
        return None
    parts = path.lstrip("/").split("/")
    if any(part in (".", "..") for part in parts):
        return None
    if path == "/":
        parts = ["index.html"]
    elif path not in ("/index.html", "/favicon.ico"):
        if len(parts) < 2 or parts[0] not in _ASSET_DIRECTORIES:
            return None
    candidate = (ui_dir / "/".join(parts)).resolve()
    if not candidate.is_relative_to(ui_dir.resolve()):
        return None
    if candidate.suffix not in _STATIC_TYPES or not candidate.is_file():
        return None
    return candidate


def _initial_record() -> dict[str, object]:
    env = env_module.GameEnv("standard_turn_zero")
    snapshot = env.record_snapshot()
    return {
        "format": FORMAT, "version": VERSION, "rules": "capture_king",
        "variant": env.spec.name, "initial": snapshot["fen"], "actions": [],
        "result": {"outcome": 0.0, "terminal_reason": "aborted"},
        "meta": {
            "iteration": 0, "game_id": 0, "backend": "native_ui",
            "board_limit": env.board_limit,
            "material_scale": env.material_scale,
        },
    }


def _preflight_native() -> None:
    env_module._require_binding()
    probe = env_module.GameEnv()
    record_snapshot = getattr(probe, "record_snapshot", None)
    if not callable(record_snapshot):
        raise RuntimeError("the native binding does not provide record_snapshot")
    record_snapshot()


class ReplayHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, server_address: tuple[str, int], ui_dir: Path) -> None:
        self.ui_dir = ui_dir
        super().__init__(server_address, ReplayRequestHandler)


class ReplayRequestHandler(BaseHTTPRequestHandler):
    server: ReplayHTTPServer
    server_version = "SuzeroReplayHTTP/1.0"
    sys_version = ""

    def _send_headers(self, status: int, content_type: str, length: int) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

    def _send_json(self, status: int, value: object) -> None:
        body = _json_bytes(value)
        self._send_headers(status, "application/json; charset=utf-8", len(body))
        self.wfile.write(body)

    def _send_error_json(self, status: int, message: str) -> None:
        self._send_json(status, {"error": message})

    def _validate_request_source(self) -> bool:
        hosts = self.headers.get_all("Host", [])
        if len(hosts) != 1:
            self._send_error_json(400, "exactly one Host header is required")
            return False
        try:
            hostname = _parse_host(hosts[0], self.server.server_port)
        except ValueError as exc:
            self._send_error_json(400, str(exc))
            return False
        origins = self.headers.get_all("Origin", [])
        if len(origins) > 1:
            self._send_error_json(403, "multiple Origin headers are not allowed")
            return False
        if origins:
            try:
                _validate_origin(origins[0], hostname, self.server.server_port)
            except ValueError as exc:
                self._send_error_json(403, str(exc))
                return False
        return True

    def do_GET(self) -> None:
        if not self._validate_request_source():
            return
        if self.path == "/api/health":
            self._send_json(200, {"ok": True})
            return
        if self.path == "/api/initial":
            try:
                record = _initial_record()
                frames: list[dict[str, object]] = []
                replay_record(record, on_frame=frames.append)
                self._send_json(200, {**record, "frames": frames})
            except RuntimeError as exc:
                self._send_error_json(503, str(exc))
            return
        asset = _static_asset(self.server.ui_dir, self.path)
        if asset is None:
            self._send_error_json(404, "not found")
            return
        try:
            body = asset.read_bytes()
            if asset.name == "index.html":
                # Use the engine's real page and renderer. Only its worker
                # transport changes; standalone WASM entrypoints stay intact.
                body = body.replace(
                    b"window.worker_path = 'scripts/worker.js';",
                    b"window.worker_path = 'scripts/gn5d-worker.js'; window.nativeReplay = true;",
                )
        except OSError:
            _LOGGER.exception("failed to read UI asset %s", asset.name)
            self._send_error_json(500, "internal server error")
            return
        self._send_headers(200, _STATIC_TYPES[asset.suffix], len(body))
        self.wfile.write(body)

    def do_POST(self) -> None:
        if not self._validate_request_source():
            return
        if self.path != "/api/replay":
            self._send_error_json(404, "not found")
            return
        if self.headers.get_all("Transfer-Encoding", []):
            self._send_error_json(400, "Transfer-Encoding is not supported")
            return
        lengths = self.headers.get_all("Content-Length", [])
        if not lengths:
            self._send_error_json(411, "Content-Length is required")
            return
        if len(lengths) != 1:
            self._send_error_json(400, "exactly one Content-Length header is required")
            return
        raw_length = lengths[0]
        if not raw_length.isascii() or not raw_length.isdigit():
            self._send_error_json(400, "invalid Content-Length")
            return
        normalized_length = raw_length.lstrip("0") or "0"
        if len(normalized_length) > len(str(_MAX_REQUEST_BYTES)):
            self._send_error_json(413, "request body exceeds the 16 MiB limit")
            return
        length = int(normalized_length)
        if length > _MAX_REQUEST_BYTES:
            self._send_error_json(413, "request body exceeds the 16 MiB limit")
            return
        body = self.rfile.read(length)
        if len(body) != length:
            self._send_error_json(400, "request body is shorter than Content-Length")
            return
        try:
            text = body.decode("utf-8")
            record = json.loads(text, parse_constant=_reject_json_constant)
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
            self._send_error_json(400, f"invalid JSON: {exc}")
            return

        frames: list[dict[str, object]] = []
        frames_bytes = 0

        def collect_frame(frame: dict[str, object]) -> None:
            nonlocal frames_bytes
            frames_bytes += len(_json_bytes(frame))
            if frames_bytes > _MAX_FRAMES_BYTES:
                raise ValueError("replay frames exceed the 128 MiB limit")
            frames.append(frame)

        try:
            replay_record(record, on_frame=collect_frame)
            response = {
                "variant": record["variant"],
                "initial": record["initial"],
                "result": record["result"],
                "meta": record["meta"],
                "actions": record["actions"],
                "frames": frames,
            }
            response_body = _json_bytes(response)
        except ValueError as exc:
            self._send_error_json(400, str(exc))
            return
        except RuntimeError as exc:
            self._send_error_json(503, str(exc))
            return
        except Exception:
            _LOGGER.exception("unexpected error while replaying a game record")
            self._send_error_json(500, "internal server error")
            return
        self._send_headers(200, "application/json; charset=utf-8", len(response_body))
        self.wfile.write(response_body)

    def log_message(self, format: str, *args: object) -> None:
        _LOGGER.info("%s - %s", self.client_address[0], format % args)


def _port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("port must be an integer") from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m suzero_5dchess.ui_server")
    parser.add_argument("--port", type=_port, default=8080)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ui_dir = _project_ui_dir()
    try:
        _check_ui_dir(ui_dir)
        _preflight_native()
    except (OSError, RuntimeError) as exc:
        print(f"cannot start replay server: {exc}", file=sys.stderr)
        return 1
    try:
        server = ReplayHTTPServer(("127.0.0.1", args.port), ui_dir)
    except OSError as exc:
        print(f"cannot bind 127.0.0.1:{args.port}: {exc}", file=sys.stderr)
        return 1
    print(f"Serving replay UI at http://127.0.0.1:{args.port}/", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
