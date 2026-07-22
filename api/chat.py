"""POST /v1/chat/completions — OpenAI-compatible, streaming SSE."""
from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

from qwen import (
    UpstreamError, collapse_messages, consume_sse, create_chat,
    extract_delta, get_config, open_completion_stream, resolve_model,
    run_with_failover, build_tool_instructions, fetch_with_tool_retry,
    TOOL_REMINDER, flatten_content, try_parse_tool_calls,
)

# vercel python: handler receives (request) via BaseHTTPRequestHandler
from http.server import BaseHTTPRequestHandler


class handler(BaseHTTPRequestHandler):

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length).decode() or "{}")
        except Exception:
            return self._error(400, "invalid JSON body")

        model = resolve_model(body.get("model"))
        cfg = get_config()
        tools = body.get("tools") if isinstance(body.get("tools"), list) else []
        tool_mode = bool(tools) and body.get("tool_choice") != "none"
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
            tool_mode=tool_mode,
            include_history=(not tool_mode and not chat_id and len(messages) > 1),
        )
        if tool_mode:
            content = (
                build_tool_instructions(tools, body.get("tool_choice"))
                + "\n" + content + "\n" + TOOL_REMINDER
            )

        if not content.strip():
            return self._error(400, "messages array produced empty content")

        if want_stream:
            return self._stream(body, model, content, tools, tool_mode,
                                cid, created, req_model, chat_id, parent_id, cfg)

        # non-streaming
        try:
            def work(tok):
                nonlocal chat_id, parent_id
                if not chat_id:
                    chat_id = create_chat(tok, model)
                    parent_id = None
                resp = open_completion_stream(
                    tok, chat_id, model, content,
                    parent_id=parent_id, auto_search=cfg["autoSearch"],
                )
                state = {
                    "reasoning": "", "reasoningDelta": "", "contentDelta": "",
                    "content": "", "finished": False, "usage": None,
                    "chat_id": chat_id, "last_msg_id": None, "response_id": None,
                }
                def on_event(frame):
                    extract_delta(frame, state)
                    if state["contentDelta"]:
                        state["content"] += state["contentDelta"]
                        state["contentDelta"] = ""
                consume_sse(resp, on_event)
                return state

            out = run_with_failover(work)
            next_parent = out.get("response_id") or out.get("last_msg_id")
            message = {
                "role": "assistant",
                "content": out.get("content"),
                "reasoning_content": out.get("reasoning") or None,
            }
            resp_obj = {
                "id": cid,
                "object": "chat.completion",
                "created": created,
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

    def _stream(self, body, model, content, tools, tool_mode,
                cid, created, req_model, chat_id, parent_id, cfg):
        messages = body.get("messages") or []
        try:
            def work(tok):
                nonlocal chat_id, parent_id
                if not chat_id:
                    chat_id = create_chat(tok, model)
                    parent_id = None
                resp = open_completion_stream(
                    tok, chat_id, model, content,
                    parent_id=parent_id, auto_search=cfg["autoSearch"],
                )
                # send SSE headers
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.send_header("X-Accel-Buffering", "no")
                self.send_header("Access-Control-Allow-Origin", "*")
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
                    "chat_id": chat_id, "last_msg_id": None, "response_id": None,
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

                next_parent = state.get("response_id") or state.get("last_msg_id")
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
            self._error(502, e.detail or e.kind)
        except Exception as e:
            try:
                self._error(502, str(e))
            except Exception:
                pass

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.end_headers()

    def _json(self, status, obj):
        raw = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(raw)

    def _error(self, status, message):
        self._json(status, {"error": {"message": message, "type": "server_error", "code": None}})

    def log_message(self, fmt, *args):
        pass
