#!/usr/bin/env python3
"""
Проверка, что WebSocket-сервер DataCode возвращает SQLite при --build_model.

Ожидается запуск сервера примерно так:
  datacode --websocket --host 0.0.0.0 --port 8899 --use-ve --build_model

Зависимость:
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
        raise AssertionError(f"sqlite_db не является валидной base64-строкой: {e}") from e


def _is_sqlite_bytes(buf: bytes) -> bool:
    return len(buf) >= len(SQLITE_MAGIC) and buf[: len(SQLITE_MAGIC)] == SQLITE_MAGIC


async def run_test(uri: str) -> None:
    # Минимальный код, создающий хотя бы одну глобальную таблицу.
    # Это нужно, чтобы сервер (при --build_model) экспортировал таблицы в SQLite.
    # Важно: используем синтаксис как в репозиторных примерах, чтобы гарантировать
    # создание Value::Table в глобалах VM.
    dc_code = "\n".join(
        [
            "global users = table([",
            '    [1, "Алиса", "Разработка"],',
            '    [2, "Боб", "Маркетинг"],',
            '    [3, "Чарли", "Разработка"],',
            '    [4, "Диана", "Продажи"]',
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
        raise AssertionError(f"Ответ не JSON: {e}. Raw={raw!r}") from e

    _assert(resp.get("success") is True, f"Ожидали success=true, получили: {resp!r}")
    _assert("output" in resp, f"В ответе нет поля output: {resp!r}")
    _assert(resp.get("error") in (None, ""), f"Ожидали error=null, получили: {resp!r}")

    sqlite_db_b64: Optional[str] = resp.get("sqlite_db")
    _assert(
        isinstance(sqlite_db_b64, str) and sqlite_db_b64.strip() != "",
        "Нет поля sqlite_db (или оно пустое). "
        "Проверьте, что сервер запущен с --build_model и что код создал global-таблицы. "
        f"Ответ сервера: {resp!r}",
    )

    sqlite_bytes = _decode_sqlite_db_field(sqlite_db_b64)
    _assert(
        _is_sqlite_bytes(sqlite_bytes),
        f"sqlite_db декодируется, но не похоже на SQLite (нет magic header). "
        f"Первые 32 байта: {sqlite_bytes[:32]!r}",
    )

    out_path = os.environ.get("DATACODE_WS_SQLITE_OUT")
    if out_path:
        with open(out_path, "wb") as f:
            f.write(sqlite_bytes)
        print(f"✅ OK: sqlite_db получен и записан в {out_path} ({len(sqlite_bytes)} bytes)")
    else:
        print(f"✅ OK: sqlite_db получен ({len(sqlite_bytes)} bytes)")


def main() -> None:
    uri = os.environ.get("DATACODE_WS_URI", "ws://127.0.0.1:8899")
    asyncio.run(run_test(uri))


if __name__ == "__main__":
    main()

