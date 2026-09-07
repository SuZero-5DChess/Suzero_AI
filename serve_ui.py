#!/usr/bin/env python3
"""
Dual-stack HTTP server (IPv4 + IPv6) for serving training UI.

Provides:
  - Static file serving from the repo root (same as before)
  - API endpoint /api/training/tail for live training log polling
  - COOP/COEP headers for SharedArrayBuffer / WASM threading support

Usage:
  python serve_ui.py [port]
"""

import os
import sys
import json
import socket
import socketserver
import http.server
import urllib.parse


# ---------------------------------------------------------------------------
# API handlers
# ---------------------------------------------------------------------------

def handle_training_tail(path: str, query: dict) -> tuple[int, dict, str]:
    """
    Read a JSONL training log file and return entries after the given iteration.

    Query params:
      path:  Relative path to the JSONL log file (e.g. alphazero/logs/very_small/training_log.jsonl)
      after: Return only entries with iteration > this value (default: 0)

    Returns: (status_code, response_dict, content_type)
    """
    log_path = query.get("path", [""])[0]
    if not log_path:
        return 400, {"error": "Missing 'path' query parameter"}, "application/json"

    after_str = query.get("after", ["0"])[0]
    try:
        after_iter = int(after_str)
    except ValueError:
        after_iter = 0

    # Resolve relative to the repo root (where the server is serving from)
    abs_path = os.path.join(os.getcwd(), log_path)
    if not os.path.isfile(abs_path):
        return 200, {"entries": [], "current_iteration": 0, "total_games": 0, "total_samples": 0}, "application/json"

    entries = []
    max_iter = 0
    total_games = 0
    total_samples = 0

    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    it = entry.get("iteration", 0)
                    if it > after_iter:
                        entries.append(entry)
                    if it > max_iter:
                        max_iter = it
                        total_games = entry.get("total_games", 0)
                        total_samples = entry.get("total_samples", 0)
                except json.JSONDecodeError:
                    continue
    except (OSError, IOError):
        return 200, {"entries": [], "current_iteration": 0, "total_games": 0, "total_samples": 0}, "application/json"

    return 200, {
        "entries": entries,
        "current_iteration": max_iter,
        "total_games": total_games,
        "total_samples": total_samples,
    }, "application/json"


# ---------------------------------------------------------------------------
# Custom request handler
# ---------------------------------------------------------------------------

API_ROUTES = {
    "/api/training/tail": handle_training_tail,
}


class TrainingUIHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP request handler with API routes and COOP/COEP headers."""

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"

        # Check API routes
        if path in API_ROUTES:
            query = urllib.parse.parse_qs(parsed.query)
            handler = API_ROUTES[path]
            status, data, content_type = handler(path, query)
            self._send_json_response(status, data)
            return

        # Default: static file serving
        # Add COOP/COEP headers for SharedArrayBuffer support
        self._add_security_headers = True
        super().do_GET()

    def do_POST(self):
        """Handle POST requests. Currently unused, but reserved for future API endpoints."""
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"

        if path in API_ROUTES:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8") if content_length else "{}"
            try:
                query = json.loads(body)
            except json.JSONDecodeError:
                query = {}
            handler = API_ROUTES[path]
            status, data, content_type = handler(path, query)
            self._send_json_response(status, data)
            return

        self.send_error(405, "Method Not Allowed")

    def _send_json_response(self, status: int, data: dict):
        """Send a JSON response with security headers."""
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._add_security_headers_to_response()
        self.end_headers()
        self.wfile.write(body)

    def _add_security_headers_to_response(self):
        """Add headers needed for SharedArrayBuffer / WASM threading."""
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")

    # Override send_response to inject security headers on every response
    def send_response(self, code, message=None):
        super().send_response(code, message)
        if getattr(self, "_add_security_headers", True):
            self._add_security_headers_to_response()

    # Silence default request logging (skip API polling noise)
    def log_message(self, format, *args):
        if self.path.startswith("/api/training/tail"):
            return
        sys.stderr.write("[%s] %s - %s\n" % (
            self.log_date_time_string(),
            self.client_address[0],
            format % args,
        ))


# ---------------------------------------------------------------------------
# Dual-stack server
# ---------------------------------------------------------------------------

class DualStackServer(socketserver.TCPServer):
    address_family = socket.AF_INET6
    allow_reuse_address = True

    def server_bind(self):
        self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        return super().server_bind()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080

    # Change to the repo root directory (same directory as this script)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    handler = TrainingUIHandler
    with DualStackServer(("::", port), handler) as httpd:
        print(f"Serving at http://127.0.0.1:{port}/ (use 127.0.0.1, not localhost, on Windows)")
        print(f"Training UI: http://127.0.0.1:{port}/ui/training.html")
        print(f"API endpoint: /api/training/tail?path=...&after=...")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")


if __name__ == "__main__":
    main()