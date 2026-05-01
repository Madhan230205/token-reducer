"""Headless Language Server Protocol client (JSON-RPC over stdio)."""

from __future__ import annotations

import contextlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


def _uri_to_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        raise ValueError(f"expected file URI, got {uri!r}")
    path = unquote(parsed.path or "")
    if sys.platform == "win32" and path.startswith("/") and len(path) >= 3 and path[2] == ":":
        path = path.lstrip("/")
    return Path(path)


def _language_id_for_ext(ext: str) -> str:
    e = ext.lower()
    if not e.startswith("."):
        e = f".{e}"
    return {
        ".py": "python",
        ".ts": "typescript",
        ".tsx": "typescriptreact",
        ".js": "javascript",
        ".jsx": "javascriptreact",
        ".go": "go",
        ".rs": "rust",
    }.get(e, e.lstrip("."))


def _normalize_definition_locations(result: Any) -> list[tuple[str, dict[str, Any]]]:
    """Return list of (uri, range_dict) from textDocument/definition result."""
    if result is None:
        return []
    if isinstance(result, dict):
        if "uri" in result:
            r = result.get("range")
            return [(str(result["uri"]), r if isinstance(r, dict) else {})]
        if "targetUri" in result:
            r = result.get("targetSelectionRange") or result.get("targetRange")
            return [(str(result["targetUri"]), r if isinstance(r, dict) else {})]
        return []
    if isinstance(result, list):
        acc: list[tuple[str, dict[str, Any]]] = []
        for item in result:
            acc.extend(_normalize_definition_locations(item))
        return acc
    return []


class HeadlessLSPClient:
    """Minimal LSP client: one subprocess, sequential JSON-RPC over stdio."""

    def __init__(self, cmd: list[str], root_path: Path) -> None:
        self.root_path = root_path.resolve()
        self._cmd = list(cmd)
        self._next_id = 1
        self.process = subprocess.Popen(
            self._cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("LSP subprocess missing stdio pipes")

    def _send_raw(self, obj: dict[str, Any]) -> None:
        payload = json.dumps(obj, separators=(",", ":"))
        message = f"Content-Length: {len(payload.encode('utf-8'))}\r\n\r\n{payload}"
        assert self.process.stdin is not None
        self.process.stdin.write(message.encode("utf-8"))
        self.process.stdin.flush()

    def _read_one_message(self) -> dict[str, Any] | None:
        assert self.process.stdout is not None
        headers: dict[str, str] = {}
        while True:
            line = b""
            while not line.endswith(b"\r\n"):
                char = self.process.stdout.read(1)
                if not char:
                    return None
                line += char
            line_s = line.decode("utf-8").strip()
            if not line_s:
                break
            if ":" not in line_s:
                continue
            key, value = line_s.split(":", 1)
            headers[key.strip().lower()] = value.strip()

        try:
            content_length = int(headers.get("content-length", "0"))
        except ValueError:
            content_length = 0
        if content_length <= 0:
            return None
        body = self.process.stdout.read(content_length)
        if not body:
            return None
        return json.loads(body.decode("utf-8"))

    def _wait_for_response(self, msg_id: int) -> dict[str, Any] | None:
        while True:
            msg = self._read_one_message()
            if msg is None:
                return None
            if msg.get("id") == msg_id:
                return msg
            # Ignore notifications and responses for other ids.

    def _send(self, method: str, params: dict[str, Any] | list[Any] | None) -> int:
        msg_id = self._next_id
        self._next_id += 1
        body: dict[str, Any] = {"jsonrpc": "2.0", "id": msg_id, "method": method}
        body["params"] = params
        payload = json.dumps(body, separators=(",", ":"))
        message = f"Content-Length: {len(payload.encode('utf-8'))}\r\n\r\n{payload}"
        assert self.process.stdin is not None
        self.process.stdin.write(message.encode("utf-8"))
        self.process.stdin.flush()
        return msg_id

    def initialize(self) -> dict[str, Any] | None:
        root_uri = self.root_path.as_uri()
        msg_id = self._send(
            "initialize",
            {
                "processId": None,
                "rootUri": root_uri,
                "capabilities": {},
            },
        )
        resp = self._wait_for_response(msg_id)
        init_payload = json.dumps(
            {"jsonrpc": "2.0", "method": "initialized", "params": {}}, separators=(",", ":")
        )
        message = f"Content-Length: {len(init_payload.encode('utf-8'))}\r\n\r\n{init_payload}"
        assert self.process.stdin is not None
        self.process.stdin.write(message.encode("utf-8"))
        self.process.stdin.flush()
        return resp

    def open_file(self, file_path: Path, text: str, ext: str) -> None:
        uri = file_path.resolve().as_uri()
        self._send_raw(
            {
                "jsonrpc": "2.0",
                "method": "textDocument/didOpen",
                "params": {
                    "textDocument": {
                        "uri": uri,
                        "languageId": _language_id_for_ext(ext),
                        "version": 1,
                        "text": text,
                    }
                },
            }
        )

    def get_definition(self, file_path: Path, line: int, character: int) -> Any:
        msg_id = self._send(
            "textDocument/definition",
            {
                "textDocument": {"uri": file_path.resolve().as_uri()},
                "position": {"line": line, "character": character},
            },
        )
        resp = self._wait_for_response(msg_id)
        if not resp or "error" in resp:
            return None
        return resp.get("result")

    def shutdown(self) -> None:
        try:
            msg_id = self._send("shutdown", None)
            self._wait_for_response(msg_id)
            exit_payload = json.dumps({"jsonrpc": "2.0", "method": "exit"}, separators=(",", ":"))
            message = f"Content-Length: {len(exit_payload.encode('utf-8'))}\r\n\r\n{exit_payload}"
            if self.process.stdin:
                try:
                    self.process.stdin.write(message.encode("utf-8"))
                    self.process.stdin.flush()
                except BrokenPipeError:
                    pass
        finally:
            with contextlib.suppress(OSError):
                self.process.kill()
            self.process.wait(timeout=5)

    def definition_snippet(
        self, file_path: Path, line: int, character: int, max_lines: int = 80
    ) -> list[dict[str, Any]]:
        """Resolve definition at position and return structured snippets from disk."""
        result = self.get_definition(file_path, line, character)
        out: list[dict[str, Any]] = []
        for uri, range_dict in _normalize_definition_locations(result):
            try:
                path = _uri_to_path(uri)
            except ValueError:
                continue
            if not path.is_file():
                continue
            try:
                raw = path.read_text(encoding="utf-8")
            except OSError:
                continue
            lines = raw.splitlines()
            start = int(range_dict.get("start", {}).get("line", 0))
            end = int(range_dict.get("end", {}).get("line", start))
            end = min(end, start + max_lines - 1)
            start = max(0, start)
            snippet = "\n".join(lines[start : end + 1])
            out.append(
                {
                    "uri": uri,
                    "path": str(path),
                    "line_start": start + 1,
                    "line_end": end + 1,
                    "snippet": snippet,
                }
            )
        return out
