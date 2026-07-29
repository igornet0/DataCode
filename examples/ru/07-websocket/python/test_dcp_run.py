#!/usr/bin/env python3
"""
Запуск DCP-пакета на WebSocket-сервере DataCode.

Требуется:
  pip install websockets datacode-dcp
  (установка DCP-python из репозитория: pip install -e path/to/DCP-python)

Запуск сервера:
  datacode --websocket --host 0.0.0.0 --port 8899 --build_model
"""

from __future__ import annotations

import asyncio
import io
import json
import sys
import tempfile
from pathlib import Path

try:
    import websockets
except ImportError:
    print("Установите websockets: pip install websockets", file=sys.stderr)
    raise SystemExit(1)

try:
    from datacode_dcp import DCPEncoder
except ImportError:
    print(
        "Установите datacode-dcp: pip install -e DCP-python",
        file=sys.stderr,
    )
    raise SystemExit(1)

WS_URI = "ws://127.0.0.1:8899"


def build_demo_package() -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp) / "data"
        data_dir.mkdir()
        (data_dir / "sample.txt").write_text("hello from DCP asset\n", encoding="utf-8")

        code = """
global text = read(path("data/sample.txt"))
print("asset:", text)
print("2 + 2 =", 2 + 2)
""".strip()

        buffer = io.BytesIO()
        (
            DCPEncoder()
            .code(code)
            .metadata({"example": "ws_dcp"})
            .assets_from_dir(data_dir)
            .write(buffer)
        )
        return buffer.getvalue()


async def run_dcp(uri: str = WS_URI) -> None:
    package = build_demo_package()
    print(f"Собран DCP-пакет ({len(package)} байт, magic {package[:4]!r})")

    async with websockets.connect(uri) as websocket:
        print(f"Подключено к {uri}")
        await websocket.send(package)
        raw = await websocket.recv()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        result = json.loads(raw)

        print("Success:", result.get("success"))
        print("Output:", result.get("output", ""))
        if result.get("error"):
            print("Error:", result["error"])
        if result.get("sqlite_db"):
            print("sqlite_db: present (base64, len", len(result["sqlite_db"]), ")")

        if not result.get("success"):
            raise SystemExit(1)


def main() -> int:
    uri = sys.argv[1] if len(sys.argv) > 1 else WS_URI
    try:
        asyncio.run(run_dcp(uri))
    except ConnectionRefusedError:
        print(f"Не удалось подключиться к {uri}", file=sys.stderr)
        print("Запустите сервер: datacode --websocket --host 0.0.0.0 --port 8899", file=sys.stderr)
        return 1
    except OSError as exc:
        if getattr(exc, "errno", None) == 61:
            print(f"Не удалось подключиться к {uri}", file=sys.stderr)
            return 1
        raise
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
