"""GET /v1/models"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
from qwen import BASE, UA, get_config

from http.server import BaseHTTPRequestHandler


class handler(BaseHTTPRequestHandler):

    def do_GET(self):
        cfg = get_config()
        try:
            req = Request(
                BASE + "/api/v2/models/",
                headers={"User-Agent": UA, "Accept": "application/json", "Version": "0.2.74", "source": "web"},
                method="GET",
            )
            with urlopen(req, timeout=15) as resp:
                j = json.loads(resp.read().decode())
            raw = (j.get("data") or {}).get("data") or []
            data = {
                "object": "list",
                "data": [
                    {"id": m.get("id"), "object": "model", "created": ((m.get("info") or {}).get("created_at") or 0), "owned_by": "qwen"}
                    for m in raw
                ],
            }
        except Exception:
            data = {"object": "list", "data": [{"id": cfg["defaultModel"], "object": "model", "created": 0, "owned_by": "qwen"}]}
        self._json(200, data)

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
