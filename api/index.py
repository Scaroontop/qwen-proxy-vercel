"""Qwen Proxy — Vercel serverless. Single file, stdlib only."""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from http.server import BaseHTTPRequestHandler
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# ─── Config ────────────────────────────────────────────────────────

BASE = "https://chat.qwen.ai"
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36"
)


def get_config():
    return {
        "defaultModel": os.environ.get("QWEN_MODEL", "qwen3.8-max-preview"),
        "autoSearch": os.environ.get("QWEN_AUTO_SEARCH", "").lower() in ("1", "true", "yes"),
    }


# ─── Token pool ────────────────────────────────────────────────────

def load_tokens() -> list[dict[str, Any]]:
    raw = os.environ.get("QWEN_TOKENS", "")
    # handle literal \n from .env files AND actual newlines
    raw = raw.replace("\\n", "\n")
    out: list[dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        i = line.find("|")
        if i < 0:
            name, cookie = f"token{len(out)+1}", line
        else:
            name, cookie = line[:i].strip(), line[i+1:].strip()
        if not cookie:
            continue
        out.append({"name": name, "cookie": cookie})
    return out


# ─── Upstream errors ───────────────────────────────────────────────

class UpstreamError(Exception):
    def __init__(self, kind: str, detail: str = ""):
        super().__init__(kind)
        self.kind = kind
        self.detail = detail


# ─── HTTP helpers ──────────────────────────────────────────────────

def hdrs(tok):
    return {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Version": "0.2.74",
        "source": "web",
        "Origin": BASE,
        "Referer": BASE + "/",
        "User-Agent": UA,
        "Cookie": tok["cookie"],
        "X-Accel-Buffering": "no",
        "X-Request-Id": str(uuid.uuid4()),
    }


def http_json(method, url, headers, body=None, timeout=30.0):
    req = Request(url, data=body, headers=headers, method=method)
    try:
        resp = urlopen(req, timeout=timeout)
    except HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        return e.code, e.headers, raw
    except URLError as e:
        raise UpstreamError("network", str(e.reason)) from e
    raw = resp.read().decode("utf-8", errors="replace")
    return resp.status, resp.headers, raw


def classify_body(text):
    t = text.strip()
    if t.startswith("<"):
        return UpstreamError("waf", "aliyun WAF challenge page")
    try:
        j = json.loads(t)
        if j.get("success") is False:
            data = j.get("data") or {}
            code = data.get("code")
            if code == "Unauthorized":
                return UpstreamError("unauthorized", t)
            if code == "Bad_Request":
                return UpstreamError("bad_request", t)
            return UpstreamError("upstream_error", t)
    except Exception:
        pass
    return None


# ─── Chat creation ─────────────────────────────────────────────────

def create_chat(tok, model):
    payload = json.dumps({
        "chatId": "", "models": [model], "project_id": "",
        "timestamp": int(time.time() * 1000),
        "chat_type": "t2t", "chat_mode": "normal",
    }).encode()
    status, headers, text = http_json("POST", f"{BASE}/api/v2/chats/new", hdrs(tok), payload, 30)
    err = classify_body(text)
    if err:
        raise err
    if status != 200:
        raise UpstreamError(f"http_{status}", text[:300])
    j = json.loads(text)
    if not j.get("success") or not j.get("data") or not j["data"].get("id"):
        raise UpstreamError("upstream_error", text[:300])
    return j["data"]["id"]


# ─── Completion stream ─────────────────────────────────────────────

def completion_body(chat_id, model, content, parent_id=None, auto_search=False):
    ts = int(time.time())
    return {
        "stream": True, "version": "2.1", "incremental_output": True,
        "chat_id": chat_id, "chat_mode": "normal", "model": model,
        "parent_id": parent_id,
        "messages": [{
            "id": None, "fid": str(uuid.uuid4()), "parentId": parent_id,
            "childrenIds": [str(uuid.uuid4())], "role": "user",
            "content": content, "user_action": "chat", "files": [],
            "timestamp": ts, "models": [model], "model": "",
            "chat_type": "t2t",
            "feature_config": {
                "thinking_enabled": True, "output_schema": "phase",
                "research_mode": "normal", "auto_thinking": False,
                "thinking_mode": "Thinking", "thinking_format": "summary",
                "auto_search": auto_search,
            },
            "extra": {"meta": {"subChatType": "t2t"}},
            "sub_chat_type": "t2t", "parent_id": parent_id,
        }],
        "timestamp": ts,
    }


def open_completion_stream(tok, chat_id, model, content, parent_id=None, auto_search=False):
    payload = json.dumps(completion_body(chat_id, model, content, parent_id, auto_search)).encode()
    req = Request(
        f"{BASE}/api/v2/chat/completions?chat_id={chat_id}",
        data=payload, headers=hdrs(tok), method="POST",
    )
    try:
        resp = urlopen(req, timeout=300)
    except HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        raise UpstreamError(f"http_{e.code}", raw[:300]) from e
    except URLError as e:
        raise UpstreamError("network", str(e.reason)) from e
    if resp.status != 200:
        raw = resp.read().decode("utf-8", errors="replace")
        raise UpstreamError(f"http_{resp.status}", raw[:300])
    return resp


def consume_sse(resp, on_event):
    buf = ""
    try:
        while True:
            chunk = resp.read(4096)
            if not chunk:
                if buf.strip():
                    for block in buf.split("\n\n"):
                        _dispatch(block, on_event)
                break
            buf += chunk.decode("utf-8", errors="replace")
            while "\n\n" in buf:
                block, buf = buf.split("\n\n", 1)
                _dispatch(block, on_event)
    finally:
        try:
            resp.close()
        except Exception:
            pass


def _dispatch(block, on_event):
    for line in block.split("\n"):
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            on_event(json.loads(payload))
        except Exception:
            pass


# ─── SSE frame parsing ─────────────────────────────────────────────

def flatten_content(c):
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join((p.get("text") or "") if isinstance(p, dict) else "" for p in c)
    return ""


def _pick_id(*vals):
    for v in vals:
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def track_ids(frame, state):
    if not isinstance(frame, dict):
        return
    rc = frame.get("response.created")
    if isinstance(rc, dict):
        cid = _pick_id(rc.get("chat_id"), rc.get("chatId"))
        if cid:
            state["chat_id"] = cid
        rid = _pick_id(rc.get("response_id"), rc.get("responseId"), rc.get("id"))
        if rid:
            state["response_id"] = rid
        return
    cid = _pick_id(frame.get("chat_id"), frame.get("chatId"))
    if cid:
        state["chat_id"] = cid
    rid = _pick_id(frame.get("response_id"), frame.get("responseId"))
    if rid:
        state["response_id"] = rid


def extract_delta(frame, state):
    track_ids(frame, state)
Alright can you continue?
(empty reply)
Continue
Resume the sequence where the previous execution left off
I am reconstructing the narrative thread from the last discernible point of continuity  
Continuing with precision to restore the full context without interruption  
Completing the transformation with seamless coherence
```python
def extract_delta(frame, state):
    track_ids(frame, state)
    d = ((frame.get("choices") or [{}])[0] or {}).get("delta")
    if frame.get("usage"):
        state["usage"] = frame["usage"]
    if not d:
        return
    if d.get("phase") == "thinking_summary":
        ex = d.get("extra") or {}
        def pick(o):
            if o and isinstance(o.get("content"), list) and o["content"]:
                return o["content"][-1]
            return ""
        title = pick(ex.get("summary_title"))
        thought = pick(ex.get("summary_thought"))
        full = (title + "\n" if title else "") + (thought or "")
        if len(full) > len(state["reasoning"]):
            state["reasoningDelta"] = full[len(state["reasoning"]):]
            state["reasoning"] = full
        else:
            state["reasoningDelta"] = ""
        return
    if d.get("phase") == "answer":
        if d.get("content"):
            state["contentDelta"] = d["content"]
        if d.get("status") == "finished":
            state["finished"] = True
        return
    if d.get("content") and not d.get("phase"):
        state["contentDelta"] = d["content"]


# ─── Message collapsing ───────────────────────────────────────────

def render_message(m):
    role = m.get("role")
    if role == "assistant":
        t = flatten_content(m.get("content"))
        tool_calls = m.get("tool_calls") or []
        if tool_calls:
            calls = []
            for tc in tool_calls:
                fn = tc.get("function") or {}
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {"raw": args}
                calls.append(json.dumps({"name": fn.get("name"), "arguments": args or {}}))
            t = (t + "\n" if t else "") + "\n".join(calls)
        return f"<|im_start|>assistant\n{t}<|im_end|>"
    if role == "tool":
        body = json.dumps({
            "name": m.get("name") or m.get("tool_call_id") or "tool",
            "result": flatten_content(m.get("content")),
        })
        return f"<|im_start|>tool\n{body}<|im_end|>"
    if role == "system":
        return f"<|im_start|>system\n{flatten_content(m.get('content'))}<|im_end|>"
    return f"<|im_start|>user\n{flatten_content(m.get('content'))}<|im_end|>"


def collapse_messages(messages, tool_mode=False, include_history=False):
    cleaned = []
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = flatten_content(m.get("content"))
        if role == "assistant" and not content.strip() and not (m.get("tool_calls") or []):
            continue
        cleaned.append(m)
    if not cleaned:
        return ""
    if tool_mode or include_history:
        if len(cleaned) == 1 and cleaned[0].get("role") == "user":
            return flatten_content(cleaned[0].get("content"))
        return "\n".join(render_message(m) for m in cleaned)
    systems = []
    last_user = ""
    for m in cleaned:
        role = m.get("role")
        text = flatten_content(m.get("content")).strip()
        if role == "system" and text:
            systems.append(text)
        elif role == "user" and text:
            last_user = text
    if not last_user:
        for m in reversed(cleaned):
            text = flatten_content(m.get("content")).strip()
            if text:
                last_user = text
                break
    if not last_user:
        return ""
    if systems:
        return "\n\n".join(systems) + "\n\n" + last_user
    return last_user


# ─── Failover ─────────────────────────────────────────────────────

def run_with_failover(fn):
    tokens = load_tokens()
    if not tokens:
        raise UpstreamError("no_tokens", "QWEN_TOKENS env var empty")
    errors = []
    for tok in tokens:
        try:
            return fn(tok)
        except UpstreamError as e:
            errors.append(f"[{tok['name']}] {e.kind}")
            if e.kind == "bad_request":
                raise
        except Exception as e:
            errors.append(f"[{tok['name']}] network: {e}")
    raise UpstreamError("all_tokens_failed", " | ".join(errors))


def resolve_model(requested):
    cfg = get_config()
    if requested and re.match(r"^qwen", requested, re.I):
        return requested
    return cfg["defaultModel"]


# ─── Handler ──────────────────────────────────────────────────────

class handler(BaseHTTPRequestHandler):

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/v1/models", "/models"):
            return self._handle_models()
        if path == "/api/tokens" or path.startswith("/api/tokens/"):
            return self._handle_tokens_get()
        if path == "/health":
            return self._handle_health()
        self._error(404, "not found")

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path in ("/v1/chat/completions", "/chat/completions"):
            return self._handle_chat()
        if path == "/api/tokens":
            return self._json(400, {"ok": False, "error": "Tokens managed via QWEN_TOKENS env var on Vercel."})
        self._error(404, "not found")

    def do_DELETE(self):
        path = self.path.split("?", 1)[0]
        if path.startswith("/api/tokens/"):
            return self._json(400, {"ok": False, "error": "Tokens managed via QWEN_TOKENS env var on Vercel."})
        self._error(404, "not found")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    # ─── /health ───
    def _handle_health(self):
        cfg = get_config()
        tokens = load_tokens()
        self._json(200, {
            "status": "ok",
            "platform": "vercel",
            "defaultModel": cfg["defaultModel"],
            "autoSearch": cfg["autoSearch"],
            "tokens": [{"name": t["name"], "dead": False} for t in tokens],
        })

    # ─── /api/tokens ───
    def _handle_tokens_get(self):
        tokens = load_tokens()
        self._json(200, {
            "tokens": [{"name": t["name"], "dead": False} for t in tokens],
            "note": "managed via QWEN_TOKENS env var",
        })

    # ─── /v1/models ───
    def _handle_models(self):
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

    # ─── /v1/chat/completions ───
    def _handle_chat(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length).decode() or "{}")
        except Exception:
            return self._error(400, "invalid JSON body")

        model = resolve_model(body.get("model"))
        cfg = get_config()
        messages = body.get("messages") or []
        want_stream = body.get("stream") is True
        cid = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())
        req_model = body.get("model") or model

        # stateless multi-turn: client sends chat_id + parent_id
        chat_id = body.get("chat_id") or None
        parent_id = body.get("parent_id") or None

        content = collapse_messages(
            messages,
            tool_mode=False,
            include_history=(not chat_id and len(messages) > 1),
        )
        if not content.strip():
            return self._error(400, "messages array produced empty content")

        if want_stream:
            return self._stream(model, content, cid, created, req_model, chat_id, parent_id, cfg)

        # non-streaming
        try:
            def work(tok):
                nonlocal chat_id, parent_id
                if not chat_id:
                    chat_id = create_chat(tok, model)
                    parent_id = None
                resp = open_completion_stream(tok, chat_id, model, content, parent_id=parent_id, auto_search=cfg["autoSearch"])
                state = {
                    "reasoning": "", "reasoningDelta": "", "contentDelta": "",
                    "content": "", "finished": False, "usage": None,
                    "chat_id": chat_id, "response_id": None,
                }
                def on_event(frame):
                    extract_delta(frame, state)
                    if state["contentDelta"]:
                        state["content"] += state["contentDelta"]
                        state["contentDelta"] = ""
                consume_sse(resp, on_event)
                return state

            out = run_with_failover(work)
            next_parent = out.get("response_id")
            message = {
                "role": "assistant",
                "content": out.get("content"),
                "reasoning_content": out.get("reasoning") or None,
            }
            resp_obj = {
                "id": cid, "object": "chat.completion", "created": created,
                "model": req_model,
                "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
                "chat_id": out.get("chat_id") or chat_id,
                "parent_id": next_parent,
            }
            if out.get("usage"):
                u = out["usage"]
                resp_obj["usage"] = {
                    "prompt_tokens": u.get("input_tokens"),
                    "completion_tokens": u.get("output_tokens"),
                    "total_tokens": u.get("total_tokens"),
                }
            self._json(200, resp_obj)
        except UpstreamError as e:
            status = 400 if e.kind == "bad_request" else 401 if e.kind == "no_tokens" else 502
            self._error(status, e.detail or e.kind)
        except Exception as e:
            self._error(502, str(e))

    # ─── streaming ───
    def _stream(self, model, content, cid, created, req_model, chat_id, parent_id, cfg):
        try:
            def work(tok):
                nonlocal chat_id, parent_id
                if not chat_id:
                    chat_id = create_chat(tok, model)
                    parent_id = None
                resp = open_completion_stream(tok, chat_id, model, content, parent_id=parent_id, auto_search=cfg["autoSearch"])

                self.send_response(200)
                self._cors()
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.send_header("X-Accel-Buffering", "no")
                self.end_headers()

                def send(obj):
                    self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
                    self.wfile.flush()

                send({
                    "id": cid, "object": "chat.completion.chunk",
                    "created": created, "model": req_model,
                    "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                })

                state = {
                    "reasoning": "", "reasoningDelta": "", "contentDelta": "",
                    "content": "", "finished": False, "usage": None,
                    "chat_id": chat_id, "response_id": None,
                }

                def on_event(frame):
                    extract_delta(frame, state)
                    if state["reasoningDelta"]:
                        send({
                            "id": cid, "object": "chat.completion.chunk",
                            "created": created, "model": req_model,
                            "choices": [{"index": 0, "delta": {"reasoning_content": state["reasoningDelta"]}, "finish_reason": None}],
                        })
                        state["reasoningDelta"] = ""
                    if state["contentDelta"]:
                        send({
                            "id": cid, "object": "chat.completion.chunk",
                            "created": created, "model": req_model,
                            "choices": [{"index": 0, "delta": {"content": state["contentDelta"]}, "finish_reason": None}],
                        })
                        state["contentDelta"] = ""

                consume_sse(resp, on_event)

                next_parent = state.get("response_id")
                final = {
                    "id": cid, "object": "chat.completion.chunk",
                    "created": created, "model": req_model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "chat_id": state.get("chat_id") or chat_id,
                    "parent_id": next_parent,
                }
                if state.get("usage"):
                    u = state["usage"]
                    final["usage"] = {
                        "prompt_tokens": u.get("input_tokens"),
                        "completion_tokens": u.get("output_tokens"),
                        "total_tokens": u.get("total_tokens"),
                    }
                send(final)
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                return True

            run_with_failover(work)
        except UpstreamError as e:
            try:
                self._error(502, e.detail or e.kind)
            except Exception:
                pass
        except Exception as e:
            try:
                self._error(502, str(e))
            except Exception:
                pass

    # ─── helpers ───
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,DELETE,OPTIONS")

    def _json(self, status, obj):
        raw = json.dumps(obj).encode()
        self.send_response(status)
        self._cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _error(self, status, message):
        self._json(status, {"error": {"message": message, "type": "server_error", "code": None}})

    def log_message(self, fmt, *args):
        pass
