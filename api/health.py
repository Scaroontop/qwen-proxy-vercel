"""GET /health"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
from qwen import load_tokens, get_config

from http.server import BaseHTTPRequestHandler


class handler(BaseHTTPRequestHandler):

    def do_GET(self):
        cfg = get_config()
        tokens = load_tokens()
        self._json(200, {
            "status": "ok",
            "platform": "vercel",
            "defaultModel": cfg["defaultModel"],
            "autoSearch": cfg["autoSearch"],
            "tokens": [{"name": t["name"], "dead": False} for t in tokens],
        })

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,OPTIONS")
        self.end_headers()

    def _json(self, status, obj):
        raw = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, fmt, *args):
        pass
