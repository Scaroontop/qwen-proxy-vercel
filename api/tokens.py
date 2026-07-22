"""GET/POST/DELETE /api/tokens — read-only on Vercel (tokens live in env).
POST/DELETE return informative errors since env vars are immutable at runtime.
For a full token management UI, use Vercel KV or Blob storage."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
from qwen import load_tokens

from http.server import BaseHTTPRequestHandler


class handler(BaseHTTPRequestHandler):

    def do_GET(self):
        tokens = load_tokens()
        data = [{"name": t["name"], "dead": False} for t in tokens]
        self._json(200, {"tokens": data, "note": "tokens managed via QWEN_TOKENS env var"})

    def do_POST(self):
        self._json(400, {
            "ok": False,
            "error": "Token management via env var on Vercel. Set QWEN_TOKENS in Vercel dashboard → Settings → Environment Variables. Format: name|cookie per line.",
        })

    def do_DELETE(self):
        self._json(400, {
            "ok": False,
            "error": "Token management via env var on Vercel. Edit QWEN_TOKENS in Vercel dashboard.",
        })

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,DELETE,OPTIONS")
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
