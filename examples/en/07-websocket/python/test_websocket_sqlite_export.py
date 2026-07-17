#!/usr/bin/env python3
"""
Verify that the DataCode WebSocket server returns SQLite with --build_model.

Expected server launch:
  datacode --websocket --host 0.0.0.0 --port 8899 --use-ve --build_model

Dependency:
  pip install websockets
"""

import asyncio
import base64
import json
import os
from typing import Any, Dict, Optional

import websockets


SQLITE_MAGIC = b"SQLite format 3\x00"


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _decode_sqlite_db_field(sqlite_db_b64: str) -> bytes:
    try:
        return base64.b64decode(sqlite_db_b64, validate=True)
    except Exception as e:
        raise AssertionError(f"sqlite_db is not a valid base64 string: {e}") from e


def _is_sqlite_bytes(buf: bytes) -> bool:
    return len(buf) >= len(SQLITE_MAGIC) and buf[: len(SQLITE_MAGIC)] == SQLITE_MAGIC


async def run_test(uri: str) -> None:
    # Minimal code that creates at least one global table.
    # Needed so the server (with --build_model) exports tables to SQLite.
    # Uses repository example syntax to ensure Value::Table in VM globals.
    dc_code = "\n".join(
        [
            "global users = table([",
            '    [1, "Alice", "Development"],',
            '    [2, "Bob", "Marketing"],',
            '    [3, "Charlie", "Development"],',
            '    [4, "Diana", "Sales"]',
            '], ["id", "name", "department"])',
            "print('rows:', len(users))",
        ]
    )

    request: Dict[str, Any] = {"type": "execute", "code": dc_code}

    async with websockets.connect(uri, max_size=64 * 1024 * 1024) as ws:
        await ws.send(json.dumps(request))
        raw = await ws.recv()

    try:
        resp: Dict[str, Any] = json.loads(raw)
    except Exception as e:
        raise AssertionError(f"Response is not JSON: {e}. Raw={raw!r}") from e

    _assert(resp.get("success") is True, f"Expected success=true, got: {resp!r}")
    _assert("output" in resp, f"Response missing output field: {resp!r}")
    _assert(resp.get("error") in (None, ""), f"Expected error=null, got: {resp!r}")

    sqlite_db_b64: Optional[str] = resp.get("sqlite_db")
    _assert(
        isinstance(sqlite_db_b64, str) and sqlite_db_b64.strip() != "",
        "Missing sqlite_db field (or empty). "
        "Check server started with --build_model and code created global tables. "
        f"Server response: {resp!r}",
    )

    sqlite_bytes = _decode_sqlite_db_field(sqlite_db_b64)
    _assert(
        _is_sqlite_bytes(sqlite_bytes),
        f"sqlite_db decodes but does not look like SQLite (no magic header). "
        f"First 32 bytes: {sqlite_bytes[:32]!r}",
    )

    out_path = os.environ.get("DATACODE_WS_SQLITE_OUT")
    if out_path:
        with open(out_path, "wb") as f:
            f.write(sqlite_bytes)
        print(f"✅ OK: sqlite_db received and written to {out_path} ({len(sqlite_bytes)} bytes)")
    else:
        print(f"✅ OK: sqlite_db received ({len(sqlite_bytes)} bytes)")


def main() -> None:
    uri = os.environ.get("DATACODE_WS_URI", "ws://127.0.0.1:8899")
    asyncio.run(run_test(uri))


if __name__ == "__main__":
    main()
