"""Shared Qwen upstream logic for Vercel serverless functions."""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import (HTTPRedirectHandler, Request, build_opener, urlopen)

BASE = "https://chat.qwen.ai"

# Anti-bot / access-verification response detector. Qwen returns these
# when the session needs another challenge (CAPTCHA, risk check). Matching
# either a 401/403 status or this body lets us fail fast with a clear
# "challenge" error instead of pretending the request succeeded.
_CHALLENGE_RE = re.compile(
    r"access\s+verification|verify\s+that\s+you\s+are|"
    r"captcha|risk|please\s+complete\s+the\s+operation",
    re.IGNORECASE,
)


def detect_challenge(status: int | None, body: str) -> bool:
    """True if Qwen response signals an anti-bot challenge."""
    if status in (401, 403):
        return True
    if not body:
        return False
    return bool(_CHALLENGE_RE.search(body))


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
    # Also split on commas as fallback for single-line env vars
    if "\n" not in raw and "," in raw:
        raw = "\n".join(t.strip() for t in raw.split(","))
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
        "Timezone": time.strftime("%a %b %d %Y %H:%M:%S GMT%z (%Z)"),
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
    if detect_challenge(status, text):
        raise UpstreamError("challenge", "qwen anti-bot challenge triggered")
    if status != 200:
        raise UpstreamError(f"http_{status}", text[:300])
    j = json.loads(text)
    if not j.get("success") or not j.get("data") or not j["data"].get("id"):
        raise UpstreamError("upstream_error", text[:300])
    return j["data"]["id"]


# ─── Completion stream ─────────────────────────────────────────────

def completion_body(chat_id: str, model: str, content: str, *, parent_id: str | None = None, auto_search: bool = False, files: list | None = None) -> dict[str, Any]:
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
            "files": files or [],
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
                "tool_enabled": False,
                "plugin_enabled": False,
            },
            "extra": {"meta": {"subChatType": "t2t"}},
            "sub_chat_type": "t2t",
            "parent_id": parent_id,
        }],
        "timestamp": ts,
    }


def open_completion_stream(tok: dict[str, Any], chat_id: str, model: str, content: str, *, parent_id: str | None = None, auto_search: bool = False, files: list | None = None):
    payload = json.dumps(completion_body(chat_id, model, content, parent_id=parent_id, auto_search=auto_search, files=files)).encode()
    headers = hdrs(tok)
    req = Request(
        f"{BASE}/api/v2/chat/completions?chat_id={chat_id}",
        data=payload, headers=headers, method="POST",
    )
    try:
        resp = urlopen(req, timeout=300)
    except HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        if detect_challenge(e.code, raw):
            raise UpstreamError("challenge", "qwen anti-bot challenge triggered") from e
        raise UpstreamError(f"http_{e.code}", raw[:300]) from e
    except URLError as e:
        raise UpstreamError("network", str(e.reason)) from e
    # Qwen returns 200 even for failures sometimes — the error shows up as
    # a JSON body instead of an event-stream. Sniff the content-type and
    # reject early so callers do not parse JSON as SSE.
    ct = (resp.headers.get("Content-Type") or "").lower()
    if resp.status != 200 or "text/event-stream" not in ct:
        raw = resp.read().decode("utf-8", errors="replace")
        if detect_challenge(resp.status, raw):
            raise UpstreamError("challenge", "qwen anti-bot challenge triggered")
        raise UpstreamError(f"http_{resp.status}", raw[:300])
    return resp


# ─── Image / file upload to qwen OSS ────────────────────────────────
# Replays the qwen.ai chat upload flow observed in network captures:
#   1. POST /api/v2/files/getstsToken with {filename, filesize, filetype:"image"}
#      authenticated by the user's cookie — returns STS creds + a signed OSS
#      object URL + the bucket / region / file_id.
#   2. PUT <file_url> with the raw image bytes, signed using OSS V4 (the URL
#      already carries all required signature query params except
#      x-oss-security-token which the SDK includes as a header). Empirically
#      the URL query is sufficient for qwen's OSS bucket: just PUT the bytes
#      with the same headers ali-oss sends and OSS accepts.

import base64
import ipaddress
import socket
import urllib.parse as up

# Local imports here (consolidated with the top-level urllib.request import).

# Standard MIME -> qwen filetype buckets the web app uses
_MIME_TO_FILETYPE = {
    "image/png": "image",
    "image/jpeg": "image",
    "image/jpg": "image",
    "image/webp": "image",
    "image/gif": "image",
    "image/bmp": "image",
}


_DOC_MIME_FILETYPE = {
    "text/plain": "file",
    "text/html": "file",
    "text/markdown": "file",
    "application/json": "file",
    "text/csv": "file",
    "application/pdf": "file",
    "application/javascript": "file",
    "text/javascript": "file",
    "text/xml": "file",
    "application/xml": "file",
    "text/yaml": "file",
    "application/x-yaml": "file",
}


_EXT_TO_MIME = {
    "txt": "text/plain",
    "html": "text/html", "htm": "text/html",
    "md": "text/markdown", "markdown": "text/markdown",
    "py": "text/plain",
    "js": "application/javascript", "mjs": "application/javascript",
    "ts": "text/plain", "tsx": "text/plain",
    "json": "application/json",
    "csv": "text/csv", "tsv": "text/csv",
    "yaml": "text/yaml", "yml": "text/yaml",
    "xml": "text/xml",
    "c": "text/x-c", "cc": "text/x-c++", "cpp": "text/x-c++",
    "h": "text/x-c", "hpp": "text/x-c++",
    "java": "text/x-java",
    "go": "text/x-go",
    "rs": "text/x-rust",
    "rb": "text/plain", "sh": "text/plain", "sql": "text/plain",
    "css": "text/plain",
    "pdf": "application/pdf",
    "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
    "gif": "image/gif", "webp": "image/webp", "bmp": "image/bmp",
}


def guess_filetype(content_type, filename):
    ct = (content_type or "").lower()
    if ct in _MIME_TO_FILETYPE:
        return "image"
    if ct in _DOC_MIME_FILETYPE:
        return "file"
    ext = (filename.rsplit(".", 1)[-1] if "." in filename else "").lower()
    if ext in _EXT_TO_MIME:
        if ext in ("png", "jpg", "jpeg", "gif", "webp", "bmp"):
            return "image"
        return "file"
    return "file"


def _get_oss_token(tok: dict[str, Any], filename: str, filesize: int, filetype: str) -> dict:
    """POST /api/v2/files/getstsToken and return the parsed data dict."""
    headers = hdrs(tok)
    body = json.dumps({
        "filename": filename,
        "filesize": str(filesize),
        "filetype": filetype,
    }).encode()
    req = Request(
        f"{BASE}/api/v2/files/getstsToken",
        data=body, headers=headers, method="POST",
    )
    resp = urlopen(req, timeout=30)
    raw = resp.read().decode("utf-8", errors="replace")
    obj = json.loads(raw)
    if not obj.get("success"):
        raise UpstreamError("oss_token", raw[:300])
    return obj["data"]


def _oss_put(file_url: str, data: bytes, content_type: str, extra_headers: dict | None = None) -> None:
    """PUT raw bytes to the signed OSS URL the token issued."""
    headers = {
        "Content-Type": content_type,
        "Accept": "*/*",
        "Origin": BASE,
        "Referer": BASE + "/",
        "User-Agent": UA,
        "x-oss-content-type": content_type,
    }
    if extra_headers:
        headers.update(extra_headers)
    req = Request(file_url, data=data, headers=headers, method="PUT")
    try:
        r = urlopen(req, timeout=60)
        if r.status != 200:
            raise UpstreamError("oss_put", f"status={r.status}")
    except HTTPError as e:
        raise UpstreamError("oss_put", f"{e.code} {e.read()[:200]!r}") from e


def upload_file_to_oss(
    tok: dict[str, Any],
    file_bytes: bytes,
    filename: str,
    content_type: str,
) -> dict:
    """Upload a single file (image OR document) to qwen's OSS bucket using a
    freshly-minted STS token, and return the qwen-formatted file descriptor
    block ready to drop into the completion request's `files` array.

    Routes to qwen's `image` or `file` filetype based on the content type /
    extension. Documents become file_class:document, showType:file. Images
    become file_class:vision, showType:image (matching the web UI shape).
    """
    ft = guess_filetype(content_type, filename)
    if not filename:
        filename = f"upload.{ft}"
    # Reconcile content type if we have an extension fallback
    ct = (content_type or "").lower()
    if ct not in _MIME_TO_FILETYPE and ct not in _DOC_MIME_FILETYPE:
        ext = (filename.rsplit(".", 1)[-1] if "." in filename else "").lower()
        if ext in _EXT_TO_MIME:
            content_type = _EXT_TO_MIME[ext]
    token = _get_oss_token(tok, filename, len(file_bytes), ft)
    file_url = token["file_url"]
    file_id = token["file_id"]
    _oss_put(file_url, file_bytes, content_type)
    is_image = (ft == "image")
    ts_now = int(time.time() * 1000)
    meta = {
        "name": filename,
        "size": len(file_bytes),
        "content_type": content_type,
    }
    if not is_image:
        meta["parse_meta"] = {"parse_status": "success"}
    return {
        "type": "image" if is_image else "file",
        "file": {
            "created_at": ts_now,
            "data": {},
            "filename": filename,
            "hash": None,
            "id": file_id,
            "user_id": None,
            "meta": meta,
            "update_at": ts_now,
            "name": filename,
            "webkitRelativePath": "",
            "size": len(file_bytes),
            "type": content_type,
        },
        "id": file_id,
        "url": file_url.split("?")[0],
        "name": filename,
        "collection_name": "",
        "progress": 100,
        "status": "uploaded",
        "greenNet": "success",
        "size": len(file_bytes),
        "error": "",
        "file_type": content_type,
        "showType": "image" if is_image else "file",
        "file_class": "vision" if is_image else "document",
    }


# Backward-compat alias — older code uses `upload_image_to_oss` for images
upload_image_to_oss = upload_file_to_oss



# OpenAI-format image extraction

_DATA_URI_RE = re.compile(r"^data:([\w/+-]+);base64,(.*)$", re.DOTALL)


class _NoRedirect(HTTPRedirectHandler):
    """Raise instead of auto-following redirects so we can revalidate hops."""
    def redirect_request(self, req, fp, code, msg, hdrs, newurl):
        raise HTTPError(newurl, code, msg, hdrs, fp)


_NO_REDIRECT_OPENER = build_opener(_NoRedirect)


def _is_disallowed_host(host: str) -> bool:
    """True if a resolved hostname points at a non-public address.

    Used to block SSRF targets: loopback, RFC1918 / ULA, link-local,
    multicast, the cloud-metadata ranges (169.254.169.254 et al),
    unspecified (`0.0.0.0`), and reserved blocks. We resolve every A/AAAA
    record and reject if ANY of them is non-public — a benign hostname that
    resolves to a private IP (e.g. via attacker DNS) must still be blocked.
    """
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return True  # unresolvable → treat as disallowed (no point proxying it)
    for family, _, _, _, sockaddr in infos:
        ip_str = sockaddr[0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            return True
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_multicast or ip.is_reserved or ip.is_unspecified
                or ip.is_multicast):
            return True
        # Explicitly block the cloud-metadata / link-local well-known ranges
        # even though is_link_local usually catches them (defense in depth).
        if str(ip).startswith("169.254.") or str(ip) == "100.100.100.200":
            return True
    return False


def _fetch_url(url: str, max_bytes: int = 8 * 1024 * 1024) -> tuple[bytes, str]:
    """Fetch a *public* HTTP(S) URL and return (bytes, content-type).

    Hardening vs. the original ``urlopen(url)``-ver throwaway:

    * Scheme allowlist — only ``http``/``https``. ``file:`` and the rest of
      urllib's handlers are rejected so a caller can't ask the proxy to read
      ``file:///etc/passwd`` and feed it to the model.

    * Host resolution guard — the resolved A/AAAA records are checked against
      a private/loopback/link-local/metadata allowlist BEFORE the connection
      is opened, so redirects to internal IPs and attacker DNS rebinding
      can't pivot into the Vercel network.

    * Size cap — reads at most ``max_bytes``; oversize attachments raise a
      structured UpstreamError instead of blindly buffering.

    * Redirect revalidation — follow at most 5 redirects and re-apply the
      scheme + host checks on every hop, since the destination host can
      differ from the original.
    """
    parsed = up.urlparse(url)
    if parsed.scheme.lower() not in ("http", "https"):
        raise UpstreamError("bad_scheme", f"scheme not allowed: {parsed.scheme}")
    if not parsed.hostname:
        raise UpstreamError("bad_url", "missing host")
    current = url
    for _ in range(5):
        parsed = up.urlparse(current)
        if parsed.scheme.lower() not in ("http", "https"):
            raise UpstreamError("bad_scheme", f"redirect to non-allowed scheme: {parsed.scheme}")
        if _is_disallowed_host(parsed.hostname):
            raise UpstreamError("ssrf_blocked", f"hostname resolves to non-public address: {parsed.hostname}")
        req = Request(current, headers={"User-Agent": UA}, method="GET")
        try:
            r = _NO_REDIRECT_OPENER.open(req, timeout=20)
        except HTTPError as e:
            if 300 <= e.code < 400:
                loc = e.headers.get("Location")
                if not loc:
                    raise UpstreamError("bad_redirect", f"redirect {e.code} without Location") from e
                current = up.urljoin(current, loc)
                continue
            raise UpstreamError(f"http_{e.code}", str(e.reason)) from e
        except URLError as e:
            raise UpstreamError("network", str(e.reason)) from e
        ct = (r.headers.get("Content-Type") or "application/octet-stream").split(";")[0].strip()
        data = r.read(max_bytes + 1)
        r.close()
        if len(data) > max_bytes:
            raise UpstreamError("file_too_big", f"fetched resource exceeds {max_bytes} bytes")
        return data, ct
    raise UpstreamError("too_many_redirects", "more than 5 redirects")


_MAX_ATTACHMENTS = 20
_MAX_ATTACHMENT_BYTES = 8 * 1024 * 1024   # per file (decoded)
_MAX_AGGREGATE_BYTES = 25 * 1024 * 1024   # all files combined
_MAX_FILENAME_LEN = 256


def _decode_b64_strict(s: str) -> bytes:
    """Strict base64 decode. Rejects malformed input outright — the looser
    default silently converts ``%%`` into an empty payload and lets malformed
    uploads through. Raises :class:`UpstreamError` with kind ``bad_base64``."""
    try:
        return base64.b64decode(s, validate=True)
    except Exception as e:  # binascii.Error subclass
        raise UpstreamError("bad_base64", str(e)) from e


def _enforce_attachment(data: bytes, total: int) -> None:
    """Per-file + aggregate size enforcer. Raises UpstreamError early so we
    don't spend OSS upload quota on payloads that will be rejected downstream."""
    if len(data) > _MAX_ATTACHMENT_BYTES:
        raise UpstreamError("file_too_big",
                            f"attachment exceeds {_MAX_ATTACHMENT_BYTES} bytes")
    if total + len(data) > _MAX_AGGREGATE_BYTES:
        raise UpstreamError("file_too_big",
                            f"total attachments exceed {_MAX_AGGREGATE_BYTES} bytes")


def extract_image_inputs(messages: list[dict]) -> tuple[list[dict], list[tuple[bytes, str, str]]]:
    """Split OpenAI multi-modal messages into (text_messages, file_inputs).

    file_inputs is a list of (bytes, filename, content_type) tuples ready to
    hand to upload_file_to_oss. text_messages is the input list with file/image
    content removed (so collapse_messages gets pure text).

    Supported attachment block shapes (inside `content` arrays):
      - {"type":"image_url", "image_url":{"url": "<data-uri or http url>"}}
      - {"type":"image_url", "image_url": "<data-uri or http url>"}
      - {"type":"input_file", "filename":"foo.html", "file_data":"<data-uri>"}
      - {"type":"file", "filename":"foo.html", "file_data":"<data-uri>"}
      - {"type":"input_file", "filename":"foo.txt",
         "file_data":{"data":"<base64>", "content_type":"text/plain"}}

    Enforced caps (raise :class:`UpstreamError`):
      - up to 20 attachments per request
      - 8 MiB per decoded attachment
      - 25 MiB total decoded bytes across all attachments
      - 256-char filename length
      - strict base64 decoding (no silent tolerance of malformed input)
    """
    img_inputs = []
    text_messages = []
    total_bytes = 0
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")
        if isinstance(content, str):
            text_messages.append(msg)
            continue
        if not isinstance(content, list):
            text_messages.append(msg)
            continue
        text_parts = []
        had_image = False
        for block in content:
            if not isinstance(block, dict):
                text_parts.append(str(block))
                continue
            bt = block.get("type")
            if bt == "text":
                text_parts.append(block.get("text", ""))
            elif bt == "image_url":
                if len(img_inputs) >= _MAX_ATTACHMENTS:
                    raise UpstreamError("too_many_files",
                                        f"more than {_MAX_ATTACHMENTS} attachments")
                url = (block.get("image_url") or {}).get("url") if isinstance(block.get("image_url"), dict) else block.get("image_url")
                if not url:
                    continue
                had_image = True
                m = _DATA_URI_RE.match(url)
                if m:
                    ct = m.group(1)
                    data = _decode_b64_strict(m.group(2))
                    ext = ct.split("/")[-1].split("+")[0]
                    fn = f"image.{ext}"
                else:
                    try:
                        data, ct = _fetch_url(url)
                    except UpstreamError:
                        raise
                    except Exception as e:
                        raise UpstreamError("fetch_img", str(e))
                    fn = (url.split("/")[-1].split("?")[0]) or f"image.{ct.split('/')[-1]}"
                _enforce_attachment(data, total_bytes)
                total_bytes += len(data)
                img_inputs.append((data, fn, ct))
            elif bt in ("input_file", "file"):
                if len(img_inputs) >= _MAX_ATTACHMENTS:
                    raise UpstreamError("too_many_files",
                                        f"more than {_MAX_ATTACHMENTS} attachments")
                fn = block.get("filename") or block.get("name") or "upload.txt"
                if len(fn) > _MAX_FILENAME_LEN:
                    raise UpstreamError("bad_input_file", "filename too long")
                fd = block.get("file_data")
                if isinstance(fd, dict):
                    b64 = fd.get("data") or fd.get("contents")
                    ct_val = fd.get("content_type") or fd.get("mime_type")
                else:
                    b64 = fd
                    ct_val = None
                if not b64:
                    continue
                had_image = True
                m = _DATA_URI_RE.match(b64)
                if m:
                    ct_val = m.group(1)
                    data = _decode_b64_strict(m.group(2))
                elif ct_val:
                    data = _decode_b64_strict(b64)
                else:
                    raise UpstreamError("bad_input_file", "input_file needs base64 data URI or content_type")
                _enforce_attachment(data, total_bytes)
                total_bytes += len(data)
                img_inputs.append((data, fn, ct_val or "text/plain"))
                # If the attached file is text-like (.html/.txt/.py/.json/etc.),
                # also inline its decoded contents into the user message. qwen's
                # web chat treats `files` entries as sidebar vision attachments
                # and does NOT paste text-file contents into the prompt the model
                # sees — so without this the model hallucinates analysis of a
                # file it never actually read. Cap at 200 KB to avoid blowing
                # the prompt budget on giant files.
                _ct = (ct_val or "").lower().split(";")[0].strip()
                if _ct.startswith("text/") or _ct in _DOC_MIME_FILETYPE:
                    try:
                        decoded = data[: 200 * 1024].decode("utf-8", errors="replace")
                    except Exception:
                        decoded = ""
                    if decoded:
                        fence = chr(96) * 3
                        lang = fn.rsplit(".", 1)[-1] if "." in fn else ""
                        text_parts.append(
                            f"\n\nHere is the full contents of {fn}:\n\n"
                            f"{fence}{lang}\n{decoded}\n{fence}\n"
                            f"\n(That is the real contents of {fn}, uploaded by "
                            f"the user. Answer questions about it directly. Do "
                            f"NOT say you cannot view it. Do NOT ask the user to "
                            f"paste it.)"
                        )
        if had_image and text_parts:
            text_messages.append({"role": role, "content": "\n".join(p for p in text_parts if p)})
        elif not had_image:
            text_messages.append(msg)
    return text_messages, img_inputs


def consume_sse(resp, on_event: Callable[[dict], None]) -> None:
    """Read an SSE stream from `resp`, dispatching one parsed JSON event at a time.

    Three correctness invariants vs. the naive reader it replaced:

      1. UTF-8 is decoded with a *stateful* incremental decoder so multibyte
         characters split across read boundaries don't turn into U+FFFD and
         corrupt the JSON payload (which the old per-chunk
         `decode(errors="replace")` did).

      2. Event framing tolerates both LF and CRLF blank-line delimiters. The
         old splitter only looked for `\\n\\n`, so a strict-CRLF upstream held
         the entire body buffered until EOF and broke live streaming.

      3. `UpstreamError` (raised by extract_delta for in-stream error frames)
         and other real exceptions propagate. The old `except Exception: pass`
         around on_event swallowed upstream errors, so streaming clients saw
         empty replies and failover never advanced tokens.

    Lines within one event are joined per SSE spec (multiple `data:` lines
    concatenate), though Qwen emits one `data:` per frame so this is mostly
    defensive.
    """
    import codecs
    dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
    buf = ""
    try:
        while True:
            chunk = resp.read(4096)
            if not chunk:
                break
            buf += dec.decode(chunk)
            # Normalize CRLF/CR line endings to LF before event framing so
            # both `\n\n` and `\r\n\r\n` delimiters collapse uniformly.
            if "\r" in buf:
                buf = buf.replace("\r\n", "\n").replace("\r", "\n")
            while "\n\n" in buf:
                block, buf = buf.split("\n\n", 1)
                dispatch_sse_block(block, on_event)
        # Drain incremental decoder's tail (final multibyte char).
        tail = dec.decode(b"", final=True)
        if tail:
            buf += tail
        if "\r" in buf:
            buf = buf.replace("\r\n", "\n").replace("\r", "\n")
        # Flush any trailing event the stream closed without a blank-line
        # delimiter for — happens when the upstream clips the last chunk.
        if buf.strip():
            dispatch_sse_block(buf, on_event)
        else:
            # Even a final frame with no trailing newline must be parsed if it
            # carries a data: line. Re-check non-blank content.
            for block in buf.split("\n\n"):
                if block.strip():
                    dispatch_sse_block(block, on_event)
    finally:
        try:
            resp.close()
        except Exception:
            pass


_DISPATCH_MAX_EVENTS = 256


def dispatch_sse_block(block: str, on_event: Callable[[dict], None]) -> None:
    """Parse one SSE event block and invoke `on_event` once per JSON data frame.

    Catches only `json.JSONDecodeError` — malformed payloads from a flaky
    upstream shouldn't take down the whole stream but also shouldn't hide
    callback-raised exceptions (which include `UpstreamError`).
    """
    data_lines = []
    for line in block.split("\n"):
        if not line.startswith("data:"):
            # SSE also allows "data:value" without the space; normalize.
            if line.startswith("data"):
                line = ":" + line[4:]
            else:
                continue
        payload = line[5:].lstrip()
        if not payload or payload == "[DONE]":
            continue
        data_lines.append(payload)
    if not data_lines:
        return
    # SSE joins multiple `data:` lines with a newline to form one event.
    # Qwen emits a single JSON object per frame, so this loop is mostly a
    # safety belt against a weird upstream splitting one object across lines.
    for raw in data_lines:
        try:
            evt = json.loads(raw)
        except json.JSONDecodeError:
            # Ignore malformed bytes; the next frame is independent.
            continue
        on_event(evt)


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


# ─── Thinking-narration leak filter ─────────────────────────────────
# Qwen sometimes echoes its internal thinking phase into the answer:
#   "Assessing the request to fix index.html"
#   "I am interpreting the user's need..."
#   "I consider the context of..."
#   "My focus is on delivering..."
# These lines start with present-tense verb-ing / "I" self-narration and
# are never part of a real answer. Strip them.

_NARRATION_PREFIXES = (
    "assessing", "considering", "interpreting", "analyzing",
    "evaluating", "examining", "reviewing", "determining",
    "attempting", "navigating", "proceeding", "responding to",
    "i proceed", "i proceed with",
    "i am interpreting", "i am assessing", "i am considering",
    "i am analyzing", "i am evaluating", "i am working",
    "i am trying", "i am responding", "i am navigating",
    "i am proceeding", "i am attempting", "i am going to",
    "i am focusing", "i am determining",
    "i am checking", "i am examining", "i am reviewing",
    "i consider", "i interpret", "i assess", "i analyze",
    "i will", "i should", "i need to", "i plan to",
    "trying to", "let me", "let's",
    "my focus is", "my focus remains", "my goal is", "my approach",
    "my response is", "while i cannot",
    "the path is", "the path appears",
    "the analysis focuses", "particular attention is being paid",
    "the analysis focuses on", "careful evaluation is being",
)

_EXCUSE_PATTERNS = (
    r"can'?t\s+(reach|access|see|view)\b.*filesystem",
    r"no\s+file\s+access",
    r"no\s+\w+\s+access\s+tools?\s+(in\s+this\s+session|here|available)",
    r"tool\s+\w+\s+(does\s+not\s+exist|doesn'?t\s+exist)",
    r"paste\s+the\s+contents?\s+of",
    r"i\s+(don'?t|do\s+not|can'?t|cannot)\s+(have|see|access|view|read).*(tool|file|filesystem|local)",
    r"don'?t\s+have\s+(filesystem|file)\s+access",
    r"no\s+(Read|Bash|file\s+tools?|actual\s+tool)",
    r"i'?m\s+(in|restricted\s+to)\s+a\s+(simulated|sandbox|limited|text)",
    r"no\s+actual\s+(tool|tools|file)",
    r"working\s+within\s+(the\s+user'?s|a\s+).*(environment|directory)",
    r"respecting\s+system\s+boundaries",
    r"technical\s+details\s+are\s+involved",
    r"no\s+external\s+tools",
    r"based\s+solely\s+on\s+the\s+(text\s+)?information\s+provided",
    r"maintain\s+\w+\s+(clarity|coherence)",
    r"prior\s+interaction\s+(involving|with)",
    r"shaped\s+by\s+(the\s+need|my\s+)",
)


def strip_thinking_narration(text: str) -> str:
    if not text or not text.strip():
        return text
    lines = text.split("\n")
    kept = []
    for line in lines:
        s = line.strip()
        if not s:
            kept.append(line)
            continue
        low = s.lower()
        # Drop introspection-style narration lines (short, no code)
        if any(low.startswith(p) for p in _NARRATION_PREFIXES):
            if len(s) < 160 and "`" not in s and "{" not in s and "def " not in s:
                continue
        # Drop tool-hallucination excuse lines anywhere in the text
        if any(re.search(pat, s, re.IGNORECASE) for pat in _EXCUSE_PATTERNS):
            if "`" not in s and "{" not in s:
                continue
        kept.append(line)
    out = "\n".join(kept)
    out = re.sub(r"\n{3,}", "\n\n", out)
    # Preserve trailing newline — caller streams lines as they arrive and
    # relies on the newline to keep adjacent lines from fusing. Final
    # whitespace-only trims are the caller's responsibility.
    return out


_TC_KEY = '"tool_calls"'


def _strip_raw_toolcalls_json(text: str) -> str:
    """Remove hallucinated raw {"tool_calls":[...]} blocks from text when the
    caller did NOT request tools. Qwen occasionally hallucinates tool-call
    JSON into the answer phase (e.g. pretending to invoke 'Bash')."""
    if not text or _TC_KEY not in text:
        return text
    out = []
    i = 0
    n = len(text)
    while i < n:
        idx = text.find(_TC_KEY, i)
        if idx < 0:
            out.append(text[i:])
            break
        # Find the enclosing object start: walk backwards to the opening '{'
        obj_start = text.rfind('{', 0, idx)
        if obj_start < 0:
            out.append(text[i:idx + len(_TC_KEY)])
            i = idx + len(_TC_KEY)
            continue
        # Find matching closing brace for the tool_calls object
        depth = 0
        j = obj_start
        in_str = False
        esc = False
        end = -1
        while j < n:
            c = text[j]
            if in_str:
                if esc:
                    esc = False
                elif c == '\\':
                    esc = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        end = j
                        break
            j += 1
        if end < 0:
            out.append(text[i:])
            break
        # Drop from obj_start to end (inclusive). Emit text before it.
        out.append(text[i:obj_start])
        i = end + 1
    return "".join(out)


def extract_delta(frame: dict, state: dict) -> None:
    track_ids(frame, state)
    # Qwen emits in-stream error events like {"error":{"code":...,...}}.
    # Raise so the upstream work fn sees a clean UpstreamError and
    # run_with_failover can advance tokens (instead of silently swallowing).
    err = frame.get("error")
    if err:
        detail = (err.get("details") or err.get("code")
                  or err.get("message") or "unknown")
        raise UpstreamError("stream_error", str(detail))
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
            _emit_answer_chunk(d["content"], state)
        if d.get("status") == "finished":
            state["finished"] = True
            # Flush any buffered partial line at end of stream
            _flush_answer_buffer(state)
        return
    if d.get("content") and not d.get("phase"):
        _emit_answer_chunk(d["content"], state)


def _emit_answer_chunk(raw: str, state: dict) -> None:
    """Buffer answer content by line, strip narration leaks, emit clean lines.

    Only writes the delta to ``state["contentDelta"]`` — accumulation of the
    full transcript into ``state["content"]`` is the *handler's* job (see the
    non-streaming and streaming on_event callbacks in api/index.py). Keeping a
    single owner for ``content`` is what prevents the double-append bug the
    naive implementation shipped with (lib appended + handler appended =
    every answer spat out twice in non-streaming mode).

    Qwen sometimes leaks its thinking-phase narration ("Assessing the request...",
    "I am interpreting...") into the answer phase. We hold content until we see a
    newline, then run strip_thinking_narration on the completed line. Partial
    trailing content (no newline) stays buffered until the next chunk or finish.
    """
    buf = state.get("answerBuf", "")
    buf += raw
    # Strip internal markers from the whole buffered text first — this lets
    # multi-chunk <|im_start|>...<|im_end|> blocks get cleaned even if the
    # opening and closing tags land in separate chunks.
    buf = re.sub(r'<\|im_start\|>.*?<\|im_end\|>', '', buf, flags=re.DOTALL)
    buf = re.sub(r'<\|tool_call\|>.*?<\|tool_call_end\|>', '', buf, flags=re.DOTALL)
    buf = re.sub(r'<\|im_start\|>', '', buf)
    buf = re.sub(r'<\|im_end\|>', '', buf)
    buf = re.sub(r'<\|tool_call\|>', '', buf)
    buf = re.sub(r'<\|tool_call_end\|>', '', buf)
    buf = re.sub(r'<\|name\>.*?</name\|>', '', buf)

    # Split off complete lines, keep the trailing partial line buffered.
    if "\n" in buf:
        idx = buf.rfind("\n")
        complete = buf[:idx + 1]
        state["answerBuf"] = buf[idx + 1:]
        cleaned = strip_thinking_narration(complete)
        if not state.get("tool_mode"):
            cleaned = _strip_raw_toolcalls_json(cleaned)
        if cleaned:
            state["contentDelta"] = (state.get("contentDelta", "") or "") + cleaned
    else:
        # No newline yet. Keep buffering up to a small cap so we can still
        # strip narration leaks; beyond that, flush live to avoid making
        # streaming feel sluggish (qwen sometimes streams answers with long
        # paragraphs and no newlines for hundreds of chars).
        state["answerBuf"] = buf
        if len(buf) > 80:
            state["answerBuf"] = ""
            flushed = buf
            if not state.get("tool_mode"):
                flushed = _strip_raw_toolcalls_json(flushed)
            if flushed:
                state["contentDelta"] = (state.get("contentDelta", "") or "") + flushed


def _flush_answer_buffer(state: dict) -> None:
    """Emit any leftover buffered content at end of stream into contentDelta.

    Only writes the delta — handlers consume it (and update ``content``).
    """
    buf = state.get("answerBuf", "")
    if not buf:
        return
    state["answerBuf"] = ""
    cleaned = strip_thinking_narration(buf)
    if not state.get("tool_mode"):
        cleaned = _strip_raw_toolcalls_json(cleaned)
    if cleaned:
        state["contentDelta"] = (state.get("contentDelta", "") or "") + cleaned


# ─── Read-only project file injection ──────────────────────────────
# Allow the served AI to "view" files in the proxy project itself. The
# server scans the last user message for a filename that matches a known
# project file and, if found, appends the file contents into the prompt
# so qwen can answer about it. Read-only: no path escapes the project root.

# Allowlist of files the served AI is allowed to "see". Path-safe keys only.
# Keep this aligned with what actually exists on disk — otherwise file-injection
# falsely matches a known alias, fails to read, and reports "couldn't open"
# instead of doing nothing.
_PROJECT_FILES = (
    "lib/qwen.py",
    "api/index.py",
    "public/index.html",
    "vercel.json",
)

_PROJECT_HINT_RE = re.compile(
    r"(?:view|see|read|show|look\s+at|fix|edit|examine|open|check|inspect|update|modify)\s+"
    r"(?:me\s+)?(?:the\s+)?(?:[\w-]+\s+){0,3}?"
    r"[`'\(\[\"<]?"
    r"(?:\.[\\/])?"
    r"(?P<path>[\w-]+(?:/[\w-]+)*\.\w{1,8})",
    re.IGNORECASE,
)

# Standalone markdown-link form [alt](path.ext) that ZCode ships when
# the user @-mentions a file. Catches bare links without an action verb.
_MARKDOWN_LINK_RE = re.compile(
    r"\[(?P<alt>[^\]]+?)\]\s*\(\s*(?:\.[\\/])?(?P<linkinner>[\w-]+(?:/[\w-]+)*\.\w{1,8})\s*\)",
    re.IGNORECASE,
)


def maybe_inject_project_file(text: str) -> str | None:
    """If the user's latest message asks to view / fix / etc a known project
    file, return the file content prefixed with a "you are reading this file"
    note. Otherwise return None.

    The returned string is meant to be appended to the user message content,
    so qwen sees the file inline and can answer questions about it directly.
    The user keeps their original text — we just tack the file on the end.
    """
    if not text:
        return None
    # First try the verb-prefixed form, then the standalone markdown-link
    # form ZCode writes when the user @-mentions a file.
    candidates = []
    for m in _PROJECT_HINT_RE.finditer(text):
        p = m.group("path")
        if p:
            candidates.append(p.strip().lstrip("./").lstrip(".\\").lower())
    for m in _MARKDOWN_LINK_RE.finditer(text):
        alt = m.group("alt") or ""
        inner = m.group("linkinner") or ""
        for cand in (inner, alt):
            if cand:
                candidates.append(cand.strip().lstrip("./").lstrip(".\\").lower())
    for candidate in candidates:
        # Match either the full path or just the basename (e.g. "index.html"
        # resolves to "public/index.html").
        for f in _PROJECT_FILES:
            if f.lower() == candidate or f.lower().endswith("/" + candidate):
                # Read the file — allowed regardless of Vercel cwd since the
                # repository is mounted at the function's working directory.
                rel = f
                tried = [
                    rel,
                    os.path.join("/var/task", rel),  # Vercel serverless root
                    os.path.join(os.getcwd(), rel),
                ]
                for path in tried:
                    try:
                        with open(path, "r", encoding="utf-8") as fh:
                            data = fh.read()
                    except (OSError, UnicodeDecodeError):
                        continue
                    if data:
                        ext = f.rsplit('.', 1)[-1] if '.' in f else ''
                        fence = chr(96) * 3
                        return (
                            f"\n\nHere is the current contents of {f}, pasted "
                            f"verbatim so I can help with it:\n\n"
                            f"{fence}{ext}\n{data}\n{fence}\n\n"
                            f"(That is the real, current {f} file from the "
                            f"project I'm working on. Answer my question "
                            f"about it directly. Do NOT refuse, do NOT say "
                            f"you cannot access files, do NOT ask me to paste "
                            f"anything. The file content is already above.)"
                        )
                # File matched but couldn't be read — acknowledge transparently
                return (
                    f"\n\nNote: {f} is in the project but the server could "
                    f"not open it in this runtime. Tell me briefly which file "
                    f"is missing, then proceed with anything else I asked."
                )
    return None


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


# ─── Message collapsing ends; tool-mode helpers below ──────────────

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
            if not isinstance(item, dict):
                continue
            fn = item.get("function")
            name = item.get("name")
            if not name and isinstance(fn, dict):
                name = fn.get("name")
            if not name:
                continue
            if known and name not in known:
                continue
            if isinstance(fn, dict):
                args = fn.get("arguments")
            else:
                args = item.get("arguments")
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
    """Return the upstream model id for a caller-supplied model string.

    Empty / None → the configured default (the "use default" path). A
    non-empty unknown id → :class:`UpstreamError` with kind ``bad_model`` so
    handlers can map it to a 400 instead of silently substituting the default
    and mislabeling the response with the caller's model name.
    """
    cfg = get_config()
    if not requested or not requested.strip():
        return cfg["defaultModel"]
    r = requested.strip().lower()
    known = {
        "qwen3.8-max-preview",
        "qwen3-235b-a22b",
        "qwen3-32b",
        "qwen3-14b",
        "qwen3-8b",
        "qwen3-next-80b-a3b",
    }
    if r in known:
        return r
    raise UpstreamError("bad_model", requested)
