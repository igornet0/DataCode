#!/usr/bin/env python3
"""Generate tests/dcp_fixtures/with_config.dcp (DCP v1.1 with __config__ fk_check)."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

DCP_MAGIC = b"DCPK"
DCP_VERSION = 0x0101
HEADER_SIZE = 64
STRING_POOL_OFFSET = 64

CODE = 'global t = table([[1, "a"]], ["id", "name"])\nprint("ok")\n'
CONFIG = b'{"fk_check":"warn"}'


def crc32(data: bytes) -> bytes:
    return struct.pack("<I", zlib.crc32(data) & 0xFFFFFFFF)


def write_string(buf: bytearray, s: str) -> None:
    b = s.encode("utf-8")
    buf.extend(struct.pack("<I", len(b)))
    buf.extend(b)


def encode_section(section_type: int, string_id: int, payload: bytes) -> bytes:
    header = bytearray()
    header.extend(struct.pack("<H", section_type))
    header.extend(struct.pack("<H", 0))
    header.extend(struct.pack("<I", string_id))
    header.extend(struct.pack("<Q", len(payload)))
    header.append(0)
    header.append(1)
    header.extend(struct.pack("<H", 0))
    header.extend(crc32(payload))
    header.extend(payload)
    return bytes(header)


def build_dcp(code: str, config: bytes) -> bytes:
    strings = ["__code__", "__config__"]
    string_pool = bytearray()
    string_pool.extend(struct.pack("<I", len(strings)))
    for s in strings:
        write_string(string_pool, s)

    code_bytes = code.encode("utf-8")
    sections = [
        (1, 0, code_bytes),
        (5, 1, config),
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
    dcp = build_dcp(CODE, CONFIG)
    out_path = Path(__file__).with_name("with_config.dcp")
    out_path.write_bytes(dcp)
    print(f"Wrote {out_path} ({len(dcp)} bytes)")


if __name__ == "__main__":
    main()
