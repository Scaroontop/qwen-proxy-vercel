"""Shared Qwen upstream logic for Vercel serverless functions."""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BASE = "https://chat.qwen.ai"
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36"
)


class UpstreamError(Exception):
    def __init__(self, kind: str, detail: str = ""):
        super().__init__(kind)
        self.kind = kind
        self.detail = detail


# ─── Token pool from env ───────────────────────────────────────────

def load_tokens() -> list[dict[str, Any]]:
    raw = os.environ.get("QWEN_TOKENS", "")
    raw = raw.replace("\\n", "\n")  # .env files store literal \n
    out: list[dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        i = line.find("|")
        if i < 0:
            name, cookie = f"token{len(out) + 1}", line
        else:
            name, cookie = line[:i].strip(), line[i + 1:].strip()
        if not cookie:
            continue
        out.append({"name": name, "cookie": cookie})
    return out

def load_tokens() -> list[dict[str, Any]]:
    raw = os.environ.get("QWEN_TOKENS", "")
    out: list[dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        i = line.find("|")
        if i < 0:
            name, cookie = f"token{len(out) + 1}", line
        else:
            name, cookie = line[:i].strip(), line[i + 1:].strip()
        if not cookie:
            continue
        out.append({"name": name, "cookie": cookie})
    return out


def get_config() -> dict[str, Any]:
    return {
        "defaultModel": os.environ.get("QWEN_MODEL", "qwen3.8-max-preview"),
        "autoSearch": os.environ.get("QWEN_AUTO_SEARCH", "").lower() in ("1", "true", "yes"),
    }


# ─── HTTP helpers ──────────────────────────────────────────────────

def hdrs(tok: dict[str, Any]) -> dict[str, str]:
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


def http_json(method: str, url: str, headers: dict[str, str], body: bytes | None = None, timeout: float = 30.0):
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


def map_code(code: str | None) -> str:
    if code == "Unauthorized":
        return "unauthorized"
    if code == "Bad_Request":
        return "bad_request"
    return "upstream_error"


def classify_body(text: str) -> UpstreamError | None:
    t = text.strip()
    if t.startswith("<"):
        return UpstreamError("waf", "aliyun WAF challenge page")
    try:
        j = json.loads(t)
        if j.get("success") is False:
            data = j.get("data") or {}
            return UpstreamError(map_code(data.get("code")), t)
    except Exception:
        pass
    return None


# ─── Chat creation ─────────────────────────────────────────────────

def create_chat(tok: dict[str, Any], model: str) -> str:
    payload = json.dumps({
        "chatId": "",
        "models": [model],
        "project_id": "",
        "timestamp": int(time.time() * 1000),
        "chat_type": "t2t",
        "chat_mode": "normal",
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

def completion_body(chat_id: str, model: str, content: str, *, parent_id: str | None = None, auto_search: bool = False) -> dict[str, Any]:
    ts = int(time.time())
    return {
        "stream": True,
        "version": "2.1",
        "incremental_output": True,
        "chat_id": chat_id,
        "chat_mode": "normal",
        "model": model,
        "parent_id": parent_id,
        "messages": [{
            "id": None,
            "fid": str(uuid.uuid4()),
            "parentId": parent_id,
            "childrenIds": [str(uuid.uuid4())],
            "role": "user",
            "content": content,
            "user_action": "chat",
            "files": [],
            "timestamp": ts,
            "models": [model],
            "model": "",
            "chat_type": "t2t",
            "feature_config": {
                "thinking_enabled": True,
                "output_schema": "phase",
                "research_mode": "normal",
                "auto_thinking": False,
                "thinking_mode": "Thinking",
                "thinking_format": "summary",
                "auto_search": auto_search,
            },
            "extra": {"meta": {"subChatType": "t2t"}},
            "sub_chat_type": "t2t",
            "parent_id": parent_id,
        }],
        "timestamp": ts,
    }


def open_completion_stream(tok: dict[str, Any], chat_id: str, model: str, content: str, *, parent_id: str | None = None, auto_search: bool = False):
    payload = json.dumps(completion_body(chat_id, model, content, parent_id=parent_id, auto_search=auto_search)).encode()
    headers = hdrs(tok)
    req = Request(
        f"{BASE}/api/v2/chat/completions?chat_id={chat_id}",
        data=payload, headers=headers, method="POST",
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


def consume_sse(resp, on_event: Callable[[dict], None]) -> None:
    buf = ""
    try:
        while True:
            chunk = resp.read(4096)
            if not chunk:
                if buf.strip():
                    for block in buf.split("\n\n"):
                        _dispatch_sse_block(block, on_event)
                break
            buf += chunk.decode("utf-8", errors="replace")
            while "\n\n" in buf:
                block, buf = buf.split("\n\n", 1)
                _dispatch_sse_block(block, on_event)
    finally:
        try:
            resp.close()
        except Exception:
            pass


def _dispatch_sse_block(block: str, on_event: Callable[[dict], None]) -> None:
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

def flatten_content(c: Any) -> str:
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join((p.get("text") or "") if isinstance(p, dict) else "" for p in c)
    return ""


def _pick_id(*vals: Any) -> str | None:
    for v in vals:
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return str(v)
    return None


def track_ids(frame: dict, state: dict) -> None:
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
            state["last_msg_id"] = rid
        return
    cid = _pick_id(
        frame.get("chat_id"), frame.get("chatId"),
        (frame.get("data") or {}).get("chat_id") if isinstance(frame.get("data"), dict) else None,
    )
    if cid:
        state["chat_id"] = cid
    rid = _pick_id(frame.get("response_id"), frame.get("responseId"))
    if rid:
        state["response_id"] = rid
        state["last_msg_id"] = rid


def extract_delta(frame: dict, state: dict) -> None:
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


# ─── Message collapsing (same logic) ──────────────────────────────

def render_message(m: dict) -> str:
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


def collapse_messages(messages: list, *, tool_mode: bool = False, include_history: bool = False) -> str:
    cleaned: list[dict] = []
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
    systems: list[str] = []
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


# ─── Tool mode helpers ─────────────────────────────────────────────

def build_tool_instructions(tools: list, tool_choice: Any) -> str:
    defs = [t.get("function") or t for t in tools]
    force = ""
    if tool_choice == "required":
        force = "You MUST call at least one tool for this request."
    elif isinstance(tool_choice, dict) and tool_choice.get("function", {}).get("name"):
        force = f'You MUST call the tool "{tool_choice["function"]["name"]}" for this request.'
    lines = [
        "<|im_start|>system",
        "You have access to the following tools, described as JSON Schema:",
        json.dumps(defs),
        "",
        "RULES for calling tools:",
        "- To call one or more tools, your ENTIRE reply must be a single raw JSON object, no other text:",
        '  {"tool_calls":[{"name":"<tool_name>","arguments":{"<arg>":"<value>"}}]}',
        "- Example: to call a tool named \"write\" with arguments filePath and content, reply with exactly:",
        '  {"tool_calls":[{"name":"write","arguments":{"filePath":"test.txt","content":"a"}}]}',
        '- Do NOT narrate, explain, or describe tool usage. Do NOT write things like "Tool X does not exist".',
        "- Do NOT wrap the JSON in markdown code fences.",
        "- Only use tools from the list above.",
        "- After you receive a message starting with <|im_start|>tool containing the result, continue normally.",
        "- If no tool is needed, answer the user normally as plain text.",
        force,
        "<|im_end|>",
    ]
    return "\n".join(x for x in lines if x is not None and x != "")


TOOL_REMINDER = (
    "<|im_start|>system\nREMINDER: To call tools, your reply must be ONLY a raw JSON object: "
    '{"tool_calls":[{"name":"<tool>","arguments":{...}}]}. No narration, no "Tool X does not exist", '
    "no markdown fences. If no tool is needed, answer in plain text.<|im_end|>"
)
TOOL_CORRECTION = (
    "\n<|im_start|>user\nYour previous reply was invalid: you narrated tool usage "
    '(e.g. "Tool X does not exist") instead of actually calling a tool. To call tools, reply with ONLY '
    'a raw JSON object like {"tool_calls":[{"name":"example","arguments":{}}]} — no other text, no markdown, '
    "no explanation. If you genuinely do not need a tool, answer in plain text.<|im_end|>"
)


def looks_like_tool_attempt(t: str | None) -> bool:
    return bool(re.search(r'\bTool\s+\w+|does not exist|<tool_call|"name"\s*:|"arguments"|```', t or ""))


def extract_json_objects(t: str) -> list[str]:
    objs = []
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, c in enumerate(t):
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
            continue
        if c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                objs.append(t[start:i + 1])
                start = -1
    return objs


def try_parse_tool_calls(text: str | None, tools: list) -> list | None:
    if not text:
        return None
    known = {(t.get("function") or t).get("name") for t in tools}
    known.discard(None)
    found = []
    for raw in extract_json_objects(text):
        try:
            j = json.loads(raw)
        except Exception:
            continue
        if isinstance(j, list):
            arr = j
        elif isinstance(j, dict) and isinstance(j.get("tool_calls"), list):
            arr = j["tool_calls"]
        elif isinstance(j, dict) and j.get("name"):
            arr = [j]
        else:
            continue
        for item in arr:
            name = item.get("name") or (item.get("function") or {}).get("name")
            if not name or (known and name not in known):
                continue
            args = item.get("arguments")
            if args is None and item.get("function"):
                args = item["function"].get("arguments")
            if args is None:
                args = {}
            if not isinstance(args, str):
                args = json.dumps(args)
            else:
                try:
                    args = json.dumps(json.loads(args))
                except Exception:
                    pass
            found.append({
                "id": "call_" + uuid.uuid4().hex[:24],
                "type": "function",
                "function": {"name": name, "arguments": args},
            })
    return found or None


# ─── Failover ──────────────────────────────────────────────────────

def run_with_failover(fn: Callable):
    tokens = load_tokens()
    if not tokens:
        raise UpstreamError("no_tokens", "QWEN_TOKENS env var empty — set it in Vercel dashboard")
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


def resolve_model(requested: str | None) -> str:
    cfg = get_config()
    if requested and re.match(r"^qwen", requested, re.I):
        return requested
    return cfg["defaultModel"]
