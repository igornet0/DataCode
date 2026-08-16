#!/usr/bin/env python3
"""
Сборка DCP из orders.csv + DataCode-скрипта и проверка SQLite (--build_model).

Требуется:
  pip install -e DCP-python
  pip install pyarrow pandas websockets

Сервер (в другом терминале):
  datacode --websocket --host 127.0.0.1 --port 8899 --build_model

Запуск:
  python3 test_orders_dcp_sqlite.py
  python3 test_orders_dcp_sqlite.py --csv /path/to/orders.csv --out orders_result.db
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import sqlite3
import sys
import tempfile
from pathlib import Path

try:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.ipc as pa_ipc
except ImportError:
    print("Установите: pip install pyarrow pandas", file=sys.stderr)
    raise SystemExit(1)

try:
    import websockets
except ImportError:
    print("Установите: pip install websockets", file=sys.stderr)
    raise SystemExit(1)

try:
    from datacode_dcp import DCPEncoder
except ImportError:
    print("Установите: pip install -e DCP-python", file=sys.stderr)
    raise SystemExit(1)

WS_URI = "ws://127.0.0.1:8899"
SQLITE_MAGIC = b"SQLite format 3\x00"

SCRIPT = """
from ws import source_table

global orders = source_table("orders", ["id", "customer_id", "product_id", "qty", "amount", "status", "region_id", "channel", "created_at"])
print("orders: ", table_info(orders))
print("-" * 50)

# Filter: Active orders
orders = orders[("status" == "active")]
print("orders: ", table_info(orders))
print("-" * 50)

# Transform: Normalize orders
orders = orders.map("amount", float)
orders = orders.rename("status", "order_status")
print("orders: ", table_info(orders))

# Formula: Order VAT
orders!["amount_vat"] = orders["amount"].map(fn(v) => v * 1.2)
print("orders: ", table_info(orders))

primary_key(orders["id"])
""".strip()


def csv_to_arrow_ipc(csv_path: Path) -> bytes:
    df = pd.read_csv(csv_path)
    table = pa.Table.from_pandas(df, preserve_index=False)
    sink = io.BytesIO()
    with pa_ipc.new_file(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue()


def build_dcp(csv_path: Path) -> bytes:
    arrow_ipc = csv_to_arrow_ipc(csv_path)
    buf = io.BytesIO()
    (
        DCPEncoder()
        .code(SCRIPT)
        .metadata({"project": "orders-pipeline-test"})
        .table("orders", arrow_ipc)
        .write(buf)
    )
    return buf.getvalue()


async def send_and_get_sqlite(uri: str, package: bytes) -> tuple[dict, bytes | None]:
    async with websockets.connect(uri, max_size=64 * 1024 * 1024) as ws:
        await ws.send(package)
        raw = await ws.recv()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        result = json.loads(raw)
        sqlite_bytes = None
        if result.get("sqlite_db"):
            sqlite_bytes = base64.b64decode(result["sqlite_db"])
        return result, sqlite_bytes


def inspect_sqlite(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    tables = [
        row[0]
        for row in cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        if not row[0].startswith("sqlite_")
    ]
    print("\n--- SQLite tables ---")
    for name in tables:
        count = cur.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
        cols = [row[1] for row in cur.execute(f'PRAGMA table_info("{name}")').fetchall()]
        print(f"  {name}: {count} rows, columns={cols}")
    if "orders" in tables:
        sample = cur.execute(
            'SELECT id, amount, order_status, amount_vat FROM orders LIMIT 3'
        ).fetchall()
        print("\n--- sample rows (orders) ---")
        for row in sample:
            print(" ", row)
    if "_datacode_schema" in tables:
        schema_rows = cur.execute("SELECT * FROM _datacode_schema").fetchall()
        print("\n--- _datacode_schema ---")
        for row in schema_rows:
            print(" ", row)
    conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).resolve().parents[4] / "orders.csv",
        help="Path to orders.csv (default: repo root orders.csv)",
    )
    parser.add_argument("--uri", default=WS_URI)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Save resulting SQLite DB to this path",
    )
    parser.add_argument(
        "--dcp-out",
        type=Path,
        default=None,
        help="Also save built .dcp package",
    )
    args = parser.parse_args()

    if not args.csv.is_file():
        print(f"CSV not found: {args.csv}", file=sys.stderr)
        return 1

    package = build_dcp(args.csv)
    print(f"DCP собран: {len(package)} bytes (magic {package[:4]!r})")

    if args.dcp_out:
        args.dcp_out.parent.mkdir(parents=True, exist_ok=True)
        args.dcp_out.write_bytes(package)
        print(f"DCP сохранён: {args.dcp_out}")

    try:
        result, sqlite_bytes = asyncio.run(send_and_get_sqlite(args.uri, package))
    except ConnectionRefusedError:
        print(
            f"Не удалось подключиться к {args.uri}\n"
            "Запустите: datacode --websocket --host 127.0.0.1 --port 8899 --build_model",
            file=sys.stderr,
        )
        return 1

    print("\n--- server response ---")
    print("success:", result.get("success"))
    if result.get("output"):
        print("output:\n", result["output"])
    if result.get("error"):
        print("error:", result["error"])

    if not result.get("success"):
        return 1

    if not sqlite_bytes or sqlite_bytes[:16] != SQLITE_MAGIC:
        print("Нет валидного sqlite_db в ответе (нужен --build_model)", file=sys.stderr)
        return 1

    out_path = args.out
    if out_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        out_path = Path(tmp.name)
        tmp.close()

    out_path.write_bytes(sqlite_bytes)
    print(f"\nSQLite сохранён: {out_path} ({len(sqlite_bytes)} bytes)")
    inspect_sqlite(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
