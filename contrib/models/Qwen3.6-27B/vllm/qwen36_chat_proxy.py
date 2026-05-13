#!/usr/bin/env python3
"""Small OpenAI-compatible guard proxy for Qwen3.6 vLLM serving.

The upstream Qwen3.6 chat template defaults to thinking mode. For this Neuron
artifact the production-safe chat path is non-thinking mode, so this proxy
injects ``chat_template_kwargs={"enable_thinking": false}`` for chat requests.
It also blocks raw completions by default because they are not chat-templated.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]):
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)


def _normalize_messages_for_qwen(messages: Any) -> Any:
    """Make common OpenAI message layouts acceptable to the Qwen chat template."""
    if not isinstance(messages, list):
        return messages

    system_parts: list[str] = []
    normal_messages: list[Any] = []
    for message in messages:
        if not isinstance(message, dict):
            normal_messages.append(message)
            continue

        role = message.get("role")
        if role in {"system", "developer"}:
            system_parts.append(_message_text(message.get("content", "")))
        else:
            normal_messages.append(message)

    if not system_parts:
        return messages

    system_message = {
        "role": "system",
        "content": "\n\n".join(part for part in system_parts if part),
    }
    return [system_message, *normal_messages]


class Qwen36ProxyHandler(BaseHTTPRequestHandler):
    backend_url: str = "http://127.0.0.1:8001"
    force_disable_thinking: bool = True
    allow_completions: bool = False

    def log_message(self, fmt: str, *args):  # noqa: D401
        print(f"{self.address_string()} - {fmt % args}", flush=True)

    def _forward(self, method: str, body: bytes | None = None):
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in {"host", "content-length", "connection"}
        }
        url = self.backend_url.rstrip("/") + self.path
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=None) as resp:
                response_body = resp.read()
                self.send_response(resp.status)
                for key, value in resp.headers.items():
                    if key.lower() in {"transfer-encoding", "connection"}:
                        continue
                    self.send_header(key, value)
                self.end_headers()
                self.wfile.write(response_body)
        except urllib.error.HTTPError as exc:
            error_body = exc.read()
            self.send_response(exc.code)
            for key, value in exc.headers.items():
                if key.lower() in {"transfer-encoding", "connection"}:
                    continue
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(error_body)

    def do_GET(self):  # noqa: N802
        self._forward("GET")

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw_body = self.rfile.read(length) if length else b""

        if self.path == "/v1/completions" and not self.allow_completions:
            _json_response(
                self,
                400,
                {
                    "error": {
                        "message": (
                            "Raw /v1/completions is disabled for Qwen3.6. "
                            "Use /v1/chat/completions so the Qwen chat template "
                            "and non-thinking mode are applied."
                        ),
                        "type": "invalid_request_error",
                        "code": "qwen36_chat_required",
                    }
                },
            )
            return

        if self.path == "/v1/chat/completions" and raw_body:
            try:
                payload = json.loads(raw_body)
            except json.JSONDecodeError:
                self._forward("POST", raw_body)
                return

            template_kwargs = payload.get("chat_template_kwargs")
            if not isinstance(template_kwargs, dict):
                template_kwargs = {}
            if self.force_disable_thinking:
                template_kwargs["enable_thinking"] = False
            else:
                template_kwargs.setdefault("enable_thinking", False)
            payload["chat_template_kwargs"] = template_kwargs
            payload["messages"] = _normalize_messages_for_qwen(payload.get("messages"))
            raw_body = json.dumps(payload).encode("utf-8")

        self._forward("POST", raw_body)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--backend-url", default=os.getenv("QWEN36_BACKEND_URL", "http://127.0.0.1:8001"))
    parser.add_argument("--allow-completions", action="store_true")
    parser.add_argument("--allow-thinking", action="store_true")
    args = parser.parse_args()

    Qwen36ProxyHandler.backend_url = args.backend_url
    Qwen36ProxyHandler.allow_completions = args.allow_completions
    Qwen36ProxyHandler.force_disable_thinking = not args.allow_thinking

    server = ThreadingHTTPServer((args.host, args.port), Qwen36ProxyHandler)
    print(
        "Qwen3.6 proxy listening on "
        f"{args.host}:{args.port}, backend={args.backend_url}, "
        f"allow_completions={args.allow_completions}, "
        f"force_disable_thinking={not args.allow_thinking}",
        flush=True,
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
