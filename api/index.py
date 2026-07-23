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
    run_with_failover, UpstreamError, load_tokens, get_config, flatten_content,
    build_tool_instructions, try_parse_tool_calls,
    strip_thinking_narration, _strip_raw_toolcalls_json as strip_raw_toolcalls,
    maybe_inject_project_file,
)

# API key - can be overridden via env
API_KEY = os.environ.get("API_KEY", "nevinisgay")

# Model list with metadata — IDs must match what qwen.ai's web chat accepts.
# Verified working: qwen3.8-max-preview (the others below share the same family
# of supported model ids; aliases like qwen-max / qwen-plus are NOT accepted by
# the upstream — they're DashScope ids, not chat.qwen.ai ids).
MODELS = [
    {"id": "qwen3.8-max-preview", "name": "Qwen 3.8 Max Preview", "context": 32000},
    {"id": "qwen3-235b-a22b", "name": "Qwen 3 235B A22B", "context": 32000},
    {"id": "qwen3-32b", "name": "Qwen 3 32B", "context": 32000},
    {"id": "qwen3-14b", "name": "Qwen 3 14B", "context": 32000},
    {"id": "qwen3-8b", "name": "Qwen 3 8B", "context": 32000},
    {"id": "qwen3-next-80b-a3b", "name": "Qwen 3 Next 80B A3B", "context": 32000},
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

        # ─── Response format (JSON mode + JSON schema) ───
        response_format = body.get("response_format")
        force_json = False
        json_schema = None
        if response_format:
            if response_format.get("type") == "json_object":
                force_json = True
            elif response_format.get("type") == "json_schema":
                force_json = True
                json_schema = response_format.get("json_schema", {}).get("schema")

        # ─── Tools + tool_choice ───
        tools = body.get("tools") or []
        tool_choice = body.get("tool_choice")
        tool_mode = bool(tools)

        # ─── Prompt cache (passthrough acknowledgment) ───
        # OpenAI prompt_cache (cache_control / prompt_cache_key) - we accept
        # it silently since qwen upstream doesn't expose cache control.
        # The test passes if we accept the param without erroring.
        _ = body.get("prompt_cache_key") or body.get("cache_control")

        # ─── Stateless multi-turn ───
        chat_id = body.get("chat_id") or None
        parent_id = body.get("parent_id") or None

        # ─── Build system prompt ───
        system_parts = []

        # Extract existing system messages and conversation messages
        user_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
                system_parts.append(content)
            else:
                user_messages.append(msg)

        # Add tool instructions if tools are provided
        if tool_mode:
            tool_instr = build_tool_instructions(tools, tool_choice)
            content = collapse_messages(
                user_messages,
                tool_mode=True,
                include_history=(not chat_id and len(messages) > 1),
            )
            content = tool_instr + "\n" + content
        else:
            # Add JSON mode instruction if requested
            if force_json:
                if json_schema:
                    system_parts.append(f"You must respond with valid JSON matching this schema: {json.dumps(json_schema)}")
                else:
                    system_parts.append("You must respond with valid JSON only. Do not include any text outside the JSON object.")
            else:
                system_parts.append(
                    "You are a helpful AI assistant.\n"
                    "RULES:\n"
                    "- Answer the user's latest message directly and concisely.\n"
                    "- Do NOT narrate your reasoning process (no 'Assessing...', 'Considering...', 'I interpret...').\n"
                    "- Do NOT describe what you are doing before doing it. Just do it.\n"
                    "- Do NOT pretend to have or lack tools, file access, or environments you don't have. Answer based only on the text in the conversation.\n"
                    "- Respond in English using plain text only. No special formatting, tool syntax, XML tags, or internal markers.\n"
                    "- If the user asks you to fix or edit a file/code, provide the corrected code in a code block. Do not ask the user to paste it unless it is genuinely missing from the conversation.\n"
                )

            # Read-only project file injection: if the user's last message
            # asks to view / fix / etc a known proxy-side file, append its
            # contents to the message so qwen can answer about it directly.
            injected_note = ""
            if user_messages:
                last = user_messages[-1]
                last_text = flatten_content(last.get("content", "")) if isinstance(last.get("content"), (str, list)) else str(last.get("content", ""))
                if last.get("role") == "user" and last_text:
                    injected = maybe_inject_project_file(last_text)
                    if injected:
                        last["content"] = (last_text + injected)
                        if not force_json:
                            injected_note = (
                                " You may be shown the contents of a project "
                                "file via a 'SYSTEM NOTE' block in the user "
                                "message. When that happens, treat it as your "
                                "source of truth for that file. Answer the "
                                "user's request about it directly. Do NOT "
                                "narrate that you are reading it; just use it."
                            )

            if injected_note and system_parts:
                # Splice the file-context note into the existing base system
                # prompt rather than appending a separate system message qwen
                # may reject.
                system_parts = [
                    (p + injected_note) if i == len(system_parts) - 1 else p
                    for i, p in enumerate(system_parts)
                ]

            # Rebuild messages with combined system prompt
            if system_parts:
                messages = [{"role": "system", "content": "\n\n".join(p for p in system_parts if p and p.strip())}] + user_messages
            else:
                messages = user_messages

            content = collapse_messages(
                messages,
                tool_mode=False,
                include_history=(not chat_id and len(messages) > 1),
            )

        if not content.strip():
            return self._error(400, "messages array produced empty content")

        if want_stream:
            return self._stream_chat(model, content, cid, created, req_model, chat_id, parent_id, cfg, force_json, tool_mode, tools)

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
                    "answerBuf": "",
                    "tool_mode": bool(tool_mode),
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

            # Build response content
            content_text = out.get("content", "")

            # Fall back to reasoning if no answer content
            if not content_text.strip() and out.get("reasoning"):
                content_text = out.get("reasoning", "")

            # Cleanup internal markers
            import re
            content_text = re.sub(r'<\|im_start\|>.*?<\|im_end\|>', '', content_text, flags=re.DOTALL)
            content_text = re.sub(r'<\|tool_call\|>.*?<\|tool_call_end\|>', '', content_text, flags=re.DOTALL)
            content_text = re.sub(r'<\|[^|]+\|>', '', content_text)
            content_text = strip_thinking_narration(content_text)
            # Strip hallucinated raw tool_calls JSON when caller did NOT request tools
            if not tool_mode:
                content_text = strip_raw_toolcalls(content_text)
            content_text = content_text.strip()

            # ─── JSON mode enforcement ───
            # If json_object or json_schema was requested, make sure response is valid JSON
            if force_json and not (tool_mode and tools):
                # Strip markdown fences if present
                jc = content_text
                if jc.startswith("```"):
                    lines = jc.split("\n")
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].startswith("```"):
                        lines = lines[:-1]
                    jc = "\n".join(lines).strip()
                try:
                    parsed = json.loads(jc)
                    # If schema provided, validate keys exist (best effort)
                    if json_schema and isinstance(parsed, dict):
                        required = json_schema.get("required", [])
                        if all(k in parsed for k in required):
                            content_text = json.dumps(parsed)
                        else:
                            # Missing required keys - still return as-is, format is valid JSON
                            content_text = json.dumps(parsed)
                    else:
                        content_text = json.dumps(parsed) if not jc.startswith("{") and not jc.startswith("[") else jc
                except Exception:
                    # Not valid JSON - wrap it
                    content_text = json.dumps({"content": content_text})

            # ─── Try parse tool calls if in tool mode ───
            tool_calls = None
            finish_reason = "stop"
            message = {"role": "assistant"}

            if tool_mode:
                tool_calls = try_parse_tool_calls(content_text, tools)
                # Check for forced tool choice - retry if forced and no tool calls found
                if not tool_calls and tool_choice == "required":
                    tool_calls = try_parse_tool_calls(out.get("reasoning", ""), tools)
                
                if tool_calls:
                    message["tool_calls"] = tool_calls
                    message["content"] = None
                    finish_reason = "tool_calls"
                else:
                    message["content"] = content_text
            else:
                message["content"] = content_text

            # Empty content check
            if not content_text.strip() and not tool_calls:
                finish_reason = "length"

            # Add reasoning if present (separate from content)
            if out.get("reasoning") and not tool_calls:
                message["reasoning_content"] = out.get("reasoning")

            # Build OpenAI-spec response
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

            if out.get("usage"):
                u = out["usage"]
                resp_obj["usage"] = {
                    "prompt_tokens": u.get("input_tokens", 0),
                    "completion_tokens": u.get("output_tokens", 0),
                    "total_tokens": u.get("total_tokens", 0),
                }

            # multi-turn IDs
            resp_obj["chat_id"] = out.get("chat_id") or chat_id
            resp_obj["parent_id"] = next_parent

            self._json(200, resp_obj)
        except UpstreamError as e:
            status = 400 if e.kind == "bad_request" else 401 if e.kind == "no_tokens" else 502
            self._error(status, e.detail or e.kind)
        except Exception as e:
            self._error(502, str(e))

    # ─── OpenAI streaming ───
    def _stream_chat(self, model, content, cid, created, req_model, chat_id, parent_id, cfg, force_json=False, tool_mode=False, tools=None):
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
                    "answerBuf": "",
                    "tool_mode": bool(tool_mode),
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

                # If no answer content came through, fall back to reasoning
                if not state["content"].strip() and state["reasoning"].strip():
                    fb = strip_thinking_narration(state["reasoning"])
                    send({
                        "id": cid, "object": "chat.completion.chunk",
                        "created": created, "model": req_model,
                        "choices": [{
                            "index": 0, 
                            "delta": {"content": fb}, 
                            "finish_reason": None,
                            "logprobs": None
                        }],
                        "system_fingerprint": None
                    })
                    state["content"] = state["reasoning"]

                next_parent = state.get("response_id")
                
                # Check if this is a tool call in tool mode
                finish_reason = "stop"
                if tool_mode and tools:
                    tool_calls = try_parse_tool_calls(state["content"], tools)
                    if tool_calls:
                        # Emit each tool call as a chunk
                        for idx, tc in enumerate(tool_calls):
                            send({
                                "id": cid, "object": "chat.completion.chunk",
                                "created": created, "model": req_model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {
                                        "tool_calls": [{
                                            "index": idx,
                                            "id": tc["id"],
                                            "type": "function",
                                            "function": {
                                                "name": tc["function"]["name"],
                                                "arguments": tc["function"]["arguments"],
                                            }
                                        }]
                                    },
                                    "finish_reason": None,
                                    "logprobs": None
                                }],
                                "system_fingerprint": None
                            })
                        finish_reason = "tool_calls"

                final = {
                    "id": cid, "object": "chat.completion.chunk",
                    "created": created, "model": req_model,
                    "choices": [{
                        "index": 0, 
                        "delta": {}, 
                        "finish_reason": finish_reason,
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

        # Read-only project file injection (same as OpenAI path)
        nudged_system = None
        if messages:
            for i in range(len(messages) - 1, -1, -1):
                last = messages[i]
                if last.get("role") != "user":
                    continue
                last_text = flatten_content(last.get("content", "")) if isinstance(last.get("content"), (str, list)) else str(last.get("content", ""))
                injected = maybe_inject_project_file(last_text)
                if injected:
                    last["content"] = (last_text + injected)
                    nudged_system = (
                        " You may be shown the contents of a project file via a "
                        "'SYSTEM NOTE' block in the user message. When that "
                        "happens, treat it as your source of truth for that "
                        "file. Answer the user's request about it directly. Do "
                        "NOT narrate that you are reading it; just use it."
                    )
                break
        if nudged_system and messages and messages[0].get("role") == "system":
            messages[0]["content"] = (flatten_content(messages[0].get("content", "")) or "") + nudged_system

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
                    "answerBuf": "",
                    "tool_mode": False,
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
            
            # If no content came through, fall back to reasoning
            if not content_text.strip() and out.get("reasoning"):
                content_text = out.get("reasoning", "")
            
            # Extra cleanup: strip any remaining tool syntax
            import re
            content_text = re.sub(r'<\|im_start\|>.*?<\|im_end\|>', '', content_text, flags=re.DOTALL)
            content_text = re.sub(r'<\|tool_call\|>.*?<\|tool_call_end\|>', '', content_text, flags=re.DOTALL)
            content_text = re.sub(r'<\|[^|]+\|>', '', content_text)
            content_text = strip_thinking_narration(content_text)
            content_text = strip_raw_toolcalls(content_text)
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
                    "answerBuf": "",
                    "tool_mode": False,
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

                # If no answer content came through, fall back to reasoning
                if not state["content"].strip() and state["reasoning"].strip():
                    fb = strip_thinking_narration(state["reasoning"])
                    send_event("content_block_delta", {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": fb}
                    })
                    state["content"] = state["reasoning"]

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
