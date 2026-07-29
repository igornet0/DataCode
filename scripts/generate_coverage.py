#!/usr/bin/env python3
"""Generate docs/ru/COVERAGE.md — builtin globals/modules vs RU docs and examples.

Source of truth: src/vm/globals.rs (BUILTIN_GLOBAL_NAMES) and src/vm/modules.rs (BUILTIN_MODULE_NAMES).
Run from repo root: python3 scripts/generate_coverage.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GLOBALS_RS = REPO / "src" / "vm" / "globals.rs"
MODULES_RS = REPO / "src" / "vm" / "modules.rs"
DOCS_RU = REPO / "docs" / "ru"
DOCS_EN = REPO / "docs" / "en"
EXAMPLES_RU = REPO / "examples" / "ru"
EXAMPLES_EN = REPO / "examples" / "en"
OUTPUT_RU = DOCS_RU / "COVERAGE.md"
OUTPUT_EN = DOCS_EN / "COVERAGE.md"

# Aliases not in BUILTIN_GLOBAL_NAMES but registered via extended_builtin_native_index
ALIASES: dict[str, str] = {
    "read": "read_file",
    "read_bin": "read_file_bin",
    "save": "save",
    "save_tables_sqlite": "save_tables_sqlite",
}

SKIP_DOC_PATHS = {OUTPUT_RU.resolve(), OUTPUT_EN.resolve()}


def parse_globals() -> list[str]:
    text = GLOBALS_RS.read_text(encoding="utf-8")
    m = re.search(
        r"pub const BUILTIN_GLOBAL_NAMES:.*?=\s*\[(.*?)\];",
        text,
        re.DOTALL,
    )
    if not m:
        sys.exit("Could not parse BUILTIN_GLOBAL_NAMES")
    return re.findall(r'"([^"]+)"', m.group(1))


def parse_modules() -> list[str]:
    text = MODULES_RS.read_text(encoding="utf-8")
    m = re.search(
        r"pub const BUILTIN_MODULE_NAMES:.*?\[(.*?)\];",
        text,
        re.DOTALL,
    )
    if not m:
        sys.exit("Could not parse BUILTIN_MODULE_NAMES")
    return re.findall(r'"([^"]+)"', m.group(1))


def collect_text_files(root: Path, suffix: str) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(root.rglob(f"*{suffix}"))


def search_names_in_file(path: Path, names: set[str]) -> set[str]:
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return set()
    found: set[str] = set()
    for name in names:
        if name in ("Table", "enum"):
            patterns = [
                rf"`{re.escape(name)}`",
                rf"\b{re.escape(name)}\s*\(",
                rf"import\s+{re.escape(name)}\b",
            ]
        else:
            patterns = [
                rf"`{re.escape(name)}`",
                rf"\b{re.escape(name)}\s*\(",
                rf"\.{re.escape(name)}\b",
                rf"\b{re.escape(name)}\b",
            ]
        for pat in patterns:
            if re.search(pat, content):
                found.add(name)
                break
    return found


def build_index(files: list[Path], names: set[str]) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {n: [] for n in names}
    for f in files:
        hits = search_names_in_file(f, names)
        rel = f.relative_to(REPO).as_posix()
        for name in hits:
            if len(index[name]) < 3:
                index[name].append(rel)
    return index


def rel_link(path: str) -> str:
    return f"[`{Path(path).name}`](../../{path})"


def main() -> None:
    globals_list = parse_globals()
    modules_list = parse_modules()
    all_names = set(globals_list) | set(modules_list) | set(ALIASES.keys())

    ru_doc_files = [
        p
        for p in collect_text_files(DOCS_RU, ".md")
        if p.resolve() not in SKIP_DOC_PATHS
    ]
    en_doc_files = [
        p
        for p in collect_text_files(DOCS_EN, ".md")
        if p.resolve() not in SKIP_DOC_PATHS
    ]
    ru_dc = collect_text_files(EXAMPLES_RU, ".dc")
    en_dc = collect_text_files(EXAMPLES_EN, ".dc")

    ru_docs_idx = build_index(ru_doc_files, all_names)
    en_docs_idx = build_index(en_doc_files, all_names)
    ru_idx = build_index(ru_dc, all_names)
    en_idx = build_index(en_dc, all_names)

    # Merge alias hits into canonical name
    for alias, canonical in ALIASES.items():
        if alias == canonical:
            continue
        for idx in (ru_docs_idx, en_docs_idx, ru_idx, en_idx):
            if alias in idx and idx[alias]:
                merged = list(dict.fromkeys(idx.get(canonical, []) + idx[alias]))[:3]
                idx[canonical] = merged

    def fmt(paths: list[str]) -> str:
        if not paths:
            return "—"
        return ", ".join(rel_link(p) for p in paths)

    def row(name: str, kind: str, docs_idx: dict, ru_ex: dict, en_ex: dict) -> str:
        d = docs_idx.get(name, [])
        r = ru_ex.get(name, [])
        e = en_ex.get(name, [])
        status = "✓" if d and (r or e) else ("doc" if d else ("ex" if (r or e) else "✗"))
        return f"| `{name}` | {kind} | {status} | {fmt(d)} | {fmt(r)} | {fmt(e)} |"

    def write_coverage(
        output: Path,
        title: str,
        intro: str,
        docs_idx: dict,
        see_also: str,
    ) -> None:
        lines = [
            title,
            "",
            intro,
            "",
            f"Source: [`globals.rs`](../../src/vm/globals.rs) — **{len(globals_list)}** globals; "
            f"[`modules.rs`](../../src/vm/modules.rs) — **{len(modules_list)}** modules.",
            "",
            "Aliases: `read` → `read_file`; examples using `read(` count toward `read_file`.",
            "",
            "| Name | Type | Status | Documentation | RU example | EN example |",
            "|------|------|--------|---------------|------------|------------|",
        ]
        for name in globals_list:
            lines.append(row(name, "global", docs_idx, ru_idx, en_idx))
        for name in modules_list:
            lines.append(row(name, "module", docs_idx, ru_idx, en_idx))
        lines.extend(
            [
                "",
                "**Status:** ✓ — doc and example; `doc` — documentation only; `ex` — example only; ✗ — gap.",
                "",
                see_also,
                "",
            ]
        )
        output.write_text("\n".join(lines), encoding="utf-8")
        print(f"Wrote {output.relative_to(REPO)} ({len(globals_list)} globals, {len(modules_list)} modules)")

    write_coverage(
        OUTPUT_RU,
        "# Покрытие: встроенные функции и модули",
        "Автогенерация: `python3 scripts/generate_coverage.py`",
        ru_docs_idx,
        "См. также: [2-язык/функции/README.md](./2-язык/функции/README.md), [2-язык/модули/README.md](./2-язык/модули/README.md).",
    )
    write_coverage(
        OUTPUT_EN,
        "# Coverage: built-in functions and modules",
        "Auto-generated: `python3 scripts/generate_coverage.py`",
        en_docs_idx,
        "See also: [2-language/functions/README.md](./2-language/functions/README.md), [2-language/modules/README.md](./2-language/modules/README.md).",
    )


if __name__ == "__main__":
    main()
