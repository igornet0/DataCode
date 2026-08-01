#!/usr/bin/env python3
"""Generate tests/dcp_fixtures/with_table.dcp (DCP v1.1 with Arrow table section)."""

from __future__ import annotations

import io
import struct
import zlib
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc

DCP_MAGIC = b"DCPK"
DCP_VERSION = 0x0101
HEADER_SIZE = 64
STRING_POOL_OFFSET = 64

CODE = (
    'from ws import source_table, tables, package_info\n'
    'print("tables:", tables())\n'
    'global orders = source_table("orders", ["id", "date", "value"])\n'
    'print("rows:", len(orders))\n'
    'print("info:", package_info())\n'
)

METADATA = {"author": "test"}


def crc32(data: bytes) -> bytes:
    return struct.pack("<I", zlib.crc32(data) & 0xFFFFFFFF)


def write_string(buf: bytearray, s: str) -> None:
    b = s.encode("utf-8")
    buf.extend(struct.pack("<I", len(b)))
    buf.extend(b)


def encode_metadata(custom: dict[str, str]) -> bytes:
    out = bytearray()
    out.extend(struct.pack("<H", 1))  # version
    out.extend(struct.pack("<I", 0))  # variable_count
    out.extend(struct.pack("<I", 0))  # resource_count
    out.extend(struct.pack("<I", len(custom)))
    for key, value in custom.items():
        write_string(out, key)
        vb = value.encode("utf-8")
        out.extend(struct.pack("<I", len(vb)))
        out.extend(vb)
    return bytes(out)


def encode_section(section_type: int, string_id: int, payload: bytes) -> bytes:
    header = bytearray()
    header.extend(struct.pack("<H", section_type))
    header.extend(struct.pack("<H", 0))  # flags
    header.extend(struct.pack("<I", string_id))
    header.extend(struct.pack("<Q", len(payload)))
    header.append(0)  # compression none
    header.append(1)  # crc32
    header.extend(struct.pack("<H", 0))  # reserved
    header.extend(crc32(payload))
    header.extend(payload)
    return bytes(header)


def build_dcp(code: str, metadata: dict[str, str], table_name: str, arrow_ipc: bytes) -> bytes:
    strings = ["__code__", "__metadata__", table_name]
    string_pool = bytearray()
    string_pool.extend(struct.pack("<I", len(strings)))
    for s in strings:
        write_string(string_pool, s)

    code_bytes = code.encode("utf-8")
    meta_bytes = encode_metadata(metadata)

    sections = [
        (1, 0, code_bytes),
        (2, 1, meta_bytes),
        (3, 2, arrow_ipc),
    ]

    encoded_sections = [encode_section(st, sid, payload) for st, sid, payload in sections]

    index_size = len(sections) * (2 + 2 + 4 + 8 + 8 + 1 + 1 + 2)
    index_offset = STRING_POOL_OFFSET + len(string_pool)
    first_section_offset = index_offset + index_size

    offsets = []
    cursor = first_section_offset
    for (_, _, payload), blob in zip(sections, encoded_sections):
        offsets.append((cursor, len(payload)))
        cursor += len(blob)

    index = bytearray()
    for (st, sid, payload), (off, dlen) in zip(sections, offsets):
        index.extend(struct.pack("<H", st))
        index.extend(struct.pack("<H", 0))
        index.extend(struct.pack("<I", sid))
        index.extend(struct.pack("<Q", off))
        index.extend(struct.pack("<Q", dlen))
        index.append(0)
        index.append(1)
        index.extend(struct.pack("<H", 0))

    package_size = cursor
    header = bytearray()
    header.extend(DCP_MAGIC)
    header.extend(struct.pack("<H", DCP_VERSION))
    header.extend(struct.pack("<H", 0))
    header.extend(struct.pack("<H", HEADER_SIZE))
    header.extend(struct.pack("<H", 0))
    header.extend(struct.pack("<I", len(sections)))
    header.extend(struct.pack("<Q", index_offset))
    header.extend(struct.pack("<Q", package_size))
    header.extend(struct.pack("<B", 1))
    header.extend(b"\x00" * (HEADER_SIZE - len(header)))

    out = bytearray()
    out.extend(header)
    out.extend(string_pool)
    out.extend(index)
    for blob in encoded_sections:
        out.extend(blob)
    return bytes(out)


def main() -> None:
    table = pa.table(
        {
            "id": [1, 2],
            "date": ["2024-01-01", "2024-01-02"],
            "value": [10.0, 20.0],
        }
    )
    sink = io.BytesIO()
    with ipc.new_file(sink, table.schema) as writer:
        writer.write_table(table)
    arrow_ipc = sink.getvalue()

    dcp = build_dcp(CODE, METADATA, "orders", arrow_ipc)
    out_path = Path(__file__).with_name("with_table.dcp")
    out_path.write_bytes(dcp)
    print(f"Wrote {out_path} ({len(dcp)} bytes)")


if __name__ == "__main__":
    main()
