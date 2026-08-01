#!/usr/bin/env python3
"""
Сборка DCP-пакета с секцией __sql__ (post-model SQL).

Требуется datacode_dcp; для --send — websockets.

Сервер:
  datacode --websocket --host 0.0.0.0 --port 8899 --build_model
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import sys

try:
    from datacode_dcp import DCPEncoder
except ImportError:
    print("Установите datacode_dcp: pip install -e DCP-python", file=sys.stderr)
    raise SystemExit(1)


CODE = """
global t = table([[1, "a"], [2, "b"]], ["id", "name"])
print("rows:", len(t))
""".strip()

GOOD_SQL = "CREATE VIEW v_names AS SELECT name FROM t ORDER BY name;"
BAD_SQL = "NOT VALID SQL;"


def build_package(bad_sql: bool = False, with_inserts: bool = True) -> bytes:
    sql = BAD_SQL if bad_sql else GOOD_SQL
    enc = DCPEncoder().code(CODE)
    if with_inserts:
        enc = enc.table_insert("t", {"id": [3, 4], "name": ["c", "d"]}).sql_table(
            'INSERT INTO "t" ("id", "name") VALUES (5, \'e\');'
        )
    enc = enc.sql(sql)
    buf = io.BytesIO()
    enc.write(buf)
    return buf.getvalue()


async def send(uri: str, package: bytes) -> None:
    import websockets

    async with websockets.connect(uri) as ws:
        await ws.send(package)
        raw = await ws.recv()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        result = json.loads(raw)
        print(json.dumps({k: result.get(k) for k in ("success", "error", "output")}, indent=2))
        if result.get("sqlite_db"):
            print("sqlite_db: есть (base64, len", len(result["sqlite_db"]), ")")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bad-sql", action="store_true")
    parser.add_argument("--send", metavar="URI")
    args = parser.parse_args()

    package = build_package(bad_sql=args.bad_sql)
    print(f"DCP собран ({len(package)} bytes)")

    if args.send:
        asyncio.run(send(args.send, package))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
