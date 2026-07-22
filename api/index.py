"""Multi-provider API proxy with OpenAI + Anthropic format support."""
from __future__ import annotations

import json
import os
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler

# Add lib to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lib.qwen import (
    create_chat, open_completion_stream, consume_sse,
    extract_delta, collapse_messages, resolve_model,
    run_with_failover, UpstreamError, load_tokens, get_config, flatten_content
)

# API key - can be overridden via env
API_KEY = os.environ.get("API_KEY", "nevinisgay")

# Model list with metadata
MODELS = [
    {"id": "qwen-max", "name": "Qwen Max", "context": 30000},
    {"id": "qwen-plus", "name": "Qwen Plus", "context": 30000},
    {"id": "qwen-turbo", "name": "Qwen Turbo", "context": 8000},
    {"id": "qwen-long", "name": "Qwen Long", "context": 1000000},
    {"id": "qwen2.5-72b-instruct", "name": "Qwen 2.5 72B", "context": 32000},
    {"id": "qwen2.5-32b-instruct", "name": "Qwen 2.5 32B", "context": 32000},
    {"id": "qwen3-235b-a22b", "name": "Qwen 3 235B", "context": 32000},
    {"id": "qwen3-32b", "name": "Qwen 3 32B", "context": 32000},
    {"id": "qwen3-14b", "name": "Qwen 3 14B", "context": 32000},
    {"id": "qwen3-8b", "name": "Qwen 3 8B", "context": 32000},
    {"id": "qwen3.8-max-preview", "name": "Qwen 3.8 Max Preview", "context": 32000},
]


class handler(BaseHTTPRequestHandler):

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/v1/models", "/models"):
            if not self._check_auth():
                return
            return self._handle_models()
        if path == "/health":
            return self._handle_health()
        if path == "/api/tokens" or path.startswith("/api/tokens/"):
            return self._handle_tokens_get()
        self._error(404, "not found")

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path in ("/v1/chat/completions", "/chat/completions"):
            if not self._check_auth():
                return
            return self._handle_chat()
        if path in ("/v1/messages",):
            if not self._check_auth():
                return
            return self._handle_messages()
        if path == "/api/tokens":
            return self._json(400, {"ok": False, "error": "Tokens managed via QWEN_TOKENS env var"})
        self._error(404, "not found")

    def do_DELETE(self):
        if self.path.startswith("/api/tokens/"):
            return self._json(400, {"ok": False, "error": "Tokens managed via QWEN_TOKENS env var"})
        self._error(404, "not found")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    # ─── Auth ───
    def _check_auth(self):
        auth = self.headers.get("Authorization", "")
        x_key = self.headers.get("x-api-key", "")
        if auth == f"Bearer {API_KEY}" or x_key == API_KEY:
            return True
        self._json(401, {
            "error": {
                "message": "Invalid API key. Use 'Authorization: Bearer <key>' or 'x-api-key: <key>'",
                "type": "auth_error",
                "code": "invalid_api_key"
            }
        })
        return False

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
        data = {
            "object": "list",
            "data": [
                {
                    "id": m["id"],
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "qwen",
                    "context_length": m.get("context", 32000)
                }
                for m in MODELS
            ]
        }
        self._json(200, data)

    # ─── /v1/chat/completions (OpenAI format) ───
    def _handle_chat(self):
        body = self._read_body()
        if body is None:
            return

        model = resolve_model(body.get("model"))
        cfg = get_config()
        messages = body.get("messages") or []
        want_stream = body.get("stream") is True
        cid = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())
        req_model = body.get("model") or model

        # Handle response_format (JSON mode)
        response_format = body.get("response_format")
        force_json = False
        json_schema = None
        if response_format:
            if response_format.get("type") == "json_object":
                force_json = True
            elif response_format.get("type") == "json_schema":
                force_json = True
                json_schema = response_format.get("json_schema", {}).get("schema")

        # Handle tools (basic support - acknowledge but don't force)
        tools = body.get("tools")
        tool_choice = body.get("tool_choice")
        
        # Stateless multi-turn
        chat_id = body.get("chat_id") or None
        parent_id = body.get("parent_id") or None

        # Build system prompt
        system_parts = []
        
        # Extract existing system messages
        user_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                system_parts.append(msg.get("content", ""))
            else:
                user_messages.append(msg)
        
        # Add JSON mode instruction if requested
        if force_json:
            if json_schema:
                system_parts.append(f"You must respond with valid JSON matching this schema: {json.dumps(json_schema)}")
            else:
                system_parts.append("You must respond with valid JSON only. Do not include any text outside the JSON object.")
        
        # Add plain text instruction (unless JSON mode)
        if not force_json:
            system_parts.append("Respond naturally in plain text. Do not use special formatting tags or internal syntax markers.")
        
        # Rebuild messages with combined system prompt
        if system_parts:
            messages = [{"role": "system", "content": "\n\n".join(system_parts)}] + user_messages
        else:
            messages = user_messages

        content = collapse_messages(
            messages,
            tool_mode=bool(tools),
            include_history=(not chat_id and len(messages) > 1),
        )
        if not content.strip():
            return self._error(400, "messages array produced empty content")

        if want_stream:
            return self._stream_chat(model, content, cid, created, req_model, chat_id, parent_id, cfg, force_json)

        # Non-streaming
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
            
            # Build proper OpenAI response
            content_text = out.get("content", "")
            
            # Extra cleanup: strip any remaining tool syntax
            import re
            content_text = re.sub(r'<\|im_start\|>.*?<\|im_end\|>', '', content_text, flags=re.DOTALL)
            content_text = re.sub(r'<\|tool_call\|>.*?<\|tool_call_end\|>', '', content_text, flags=re.DOTALL)
            content_text = re.sub(r'<\|[^|]+\|>', '', content_text)
            content_text = content_text.strip()
            
            finish_reason = "stop"
            
            # Check if content is empty (refusal case)
            if not content_text.strip():
                finish_reason = "length"
            
            message = {
                "role": "assistant",
                "content": content_text,
            }
            
            # Add reasoning if present
            if out.get("reasoning"):
                message["reasoning_content"] = out.get("reasoning")
            
            # Build response matching OpenAI spec
            resp_obj = {
                "id": cid,
                "object": "chat.completion",
                "created": created,
                "model": req_model,
                "choices": [{
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                    "logprobs": None
                }],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                },
                "system_fingerprint": None
            }
            
            # Add usage if available
            if out.get("usage"):
                u = out["usage"]
                resp_obj["usage"] = {
                    "prompt_tokens": u.get("input_tokens", 0),
                    "completion_tokens": u.get("output_tokens", 0),
                    "total_tokens": u.get("total_tokens", 0),
                }
            
            # Add multi-turn IDs (custom extension)
            resp_obj["chat_id"] = out.get("chat_id") or chat_id
            resp_obj["parent_id"] = next_parent
            
            self._json(200, resp_obj)
        except UpstreamError as e:
            status = 400 if e.kind == "bad_request" else 401 if e.kind == "no_tokens" else 502
            self._error(status, e.detail or e.kind)
        except Exception as e:
            self._error(502, str(e))

    # ─── OpenAI streaming ───
    def _stream_chat(self, model, content, cid, created, req_model, chat_id, parent_id, cfg, force_json=False):
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
                    "choices": [{
                        "index": 0, 
                        "delta": {"role": "assistant", "content": ""}, 
                        "finish_reason": None,
                        "logprobs": None
                    }],
                    "system_fingerprint": None
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
                            "choices": [{
                                "index": 0, 
                                "delta": {"reasoning_content": state["reasoningDelta"]}, 
                                "finish_reason": None,
                                "logprobs": None
                            }],
                            "system_fingerprint": None
                        })
                        state["reasoningDelta"] = ""
                    if state["contentDelta"]:
                        send({
                            "id": cid, "object": "chat.completion.chunk",
                            "created": created, "model": req_model,
                            "choices": [{
                                "index": 0, 
                                "delta": {"content": state["contentDelta"]}, 
                                "finish_reason": None,
                                "logprobs": None
                            }],
                            "system_fingerprint": None
                        })
                        state["contentDelta"] = ""

                consume_sse(resp, on_event)

                next_parent = state.get("response_id")
                final = {
                    "id": cid, "object": "chat.completion.chunk",
                    "created": created, "model": req_model,
                    "choices": [{
                        "index": 0, 
                        "delta": {}, 
                        "finish_reason": "stop",
                        "logprobs": None
                    }],
                    "system_fingerprint": None,
                    "chat_id": state.get("chat_id") or chat_id,
                    "parent_id": next_parent,
                }
                if state.get("usage"):
                    u = state["usage"]
                    final["usage"] = {
                        "prompt_tokens": u.get("input_tokens", 0),
                        "completion_tokens": u.get("output_tokens", 0),
                        "total_tokens": u.get("total_tokens", 0),
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

    # ─── /v1/messages (Anthropic format) ───
    def _handle_messages(self):
        body = self._read_body()
        if body is None:
            return

        # Convert Anthropic format to internal
        anthropic_messages = body.get("messages") or []
        system_prompt = body.get("system", "")
        
        # Convert to OpenAI-style messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        for msg in anthropic_messages:
            role = msg.get("role")
            content = msg.get("content")
            
            # Handle content array or string
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                content = "\n".join(text_parts)
            
            messages.append({"role": role, "content": content})

        model = resolve_model(body.get("model"))
        cfg = get_config()
        want_stream = body.get("stream") is True
        max_tokens = body.get("max_tokens", 4096)
        msg_id = "msg_" + uuid.uuid4().hex[:24]
        created = int(time.time())

        content = collapse_messages(messages, tool_mode=False, include_history=True)
        if not content.strip():
            return self._anthropic_error(400, "invalid_request_error", "messages array produced empty content")

        if want_stream:
            return self._stream_messages(model, content, msg_id, body.get("model") or model, cfg)

        # Non-streaming Anthropic response
        try:
            def work(tok):
                chat_id = create_chat(tok, model)
                resp = open_completion_stream(tok, chat_id, model, content, parent_id=None, auto_search=cfg["autoSearch"])
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
            
            # Build Anthropic response
            content_text = out.get("content", "")
            
            # Extra cleanup: strip any remaining tool syntax
            import re
            content_text = re.sub(r'<\|im_start\|>.*?<\|im_end\|>', '', content_text, flags=re.DOTALL)
            content_text = re.sub(r'<\|tool_call\|>.*?<\|tool_call_end\|>', '', content_text, flags=re.DOTALL)
            content_text = re.sub(r'<\|[^|]+\|>', '', content_text)
            content_text = content_text.strip()
            
            resp_obj = {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": content_text}],
                "model": body.get("model") or model,
                "stop_reason": "end_turn",
            }
            if out.get("usage"):
                u = out["usage"]
                resp_obj["usage"] = {
                    "input_tokens": u.get("input_tokens", 0),
                    "output_tokens": u.get("output_tokens", 0),
                }
            self._json(200, resp_obj)
        except UpstreamError as e:
            self._anthropic_error(502, "api_error", e.detail or e.kind)
        except Exception as e:
            self._anthropic_error(502, "api_error", str(e))

    # ─── Anthropic streaming ───
    def _stream_messages(self, model, content, msg_id, req_model, cfg):
        try:
            def work(tok):
                chat_id = create_chat(tok, model)
                resp = open_completion_stream(tok, chat_id, model, content, parent_id=None, auto_search=cfg["autoSearch"])

                self.send_response(200)
                self._cors()
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.send_header("X-Accel-Buffering", "no")
                self.end_headers()

                def send_event(event_type, data):
                    self.wfile.write(f"event: {event_type}\n".encode())
                    self.wfile.write(f"data: {json.dumps(data)}\n\n".encode())
                    self.wfile.flush()

                # message_start
                send_event("message_start", {
                    "type": "message_start",
                    "message": {
                        "id": msg_id,
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": req_model,
                        "stop_reason": None,
                        "usage": {"input_tokens": 0, "output_tokens": 0}
                    }
                })

                # content_block_start
                send_event("content_block_start", {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""}
                })

                state = {
                    "reasoning": "", "reasoningDelta": "", "contentDelta": "",
                    "content": "", "finished": False, "usage": None,
                    "chat_id": chat_id, "response_id": None,
                }
                output_tokens = 0

                def on_event(frame):
                    nonlocal output_tokens
                    extract_delta(frame, state)
                    if state["contentDelta"]:
                        send_event("content_block_delta", {
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {"type": "text_delta", "text": state["contentDelta"]}
                        })
                        output_tokens += len(state["contentDelta"]) // 4  # rough estimate
                        state["contentDelta"] = ""

                consume_sse(resp, on_event)

                # content_block_stop
                send_event("content_block_stop", {
                    "type": "content_block_stop",
                    "index": 0
                })

                # message_delta
                usage_data = {}
                if state.get("usage"):
                    u = state["usage"]
                    usage_data = {
                        "output_tokens": u.get("output_tokens", output_tokens)
                    }
                else:
                    usage_data = {"output_tokens": output_tokens}

                send_event("message_delta", {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": usage_data
                })

                # message_stop
                send_event("message_stop", {"type": "message_stop"})
                
                return True

            run_with_failover(work)
        except Exception as e:
            try:
                self._anthropic_error(502, "api_error", str(e))
            except Exception:
                pass

    # ─── Helpers ───
    def _read_body(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(length).decode() or "{}")
        except Exception:
            self._error(400, "invalid JSON body")
            return None

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

    def _anthropic_error(self, status, error_type, message):
        self._json(status, {
            "type": "error",
            "error": {
                "type": error_type,
                "message": message
            }
        })

    def log_message(self, fmt, *args):
        pass
