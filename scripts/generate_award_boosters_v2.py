#!/usr/bin/env python3
"""Run award-booster generation with configurable input/output directories."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


def _load_base_module():
    try:
        import generate_award_boosters as m  # type: ignore

        return m
    except Exception:
        p = Path(__file__).resolve().parent.parent / "archive_legacy" / "scripts" / "generate_award_boosters.py"
        if not p.exists():
            raise
        spec = importlib.util.spec_from_file_location("generate_award_boosters_legacy", p)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to load legacy module spec: {p}")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate booster artifacts with custom paths.")
    parser.add_argument("--tables-dir", required=True, help="Directory containing analysis tables csv files.")
    parser.add_argument("--output-dir", required=True, help="Run output root directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = _load_base_module()
    tables_dir = Path(args.tables_dir)
    output_dir = Path(args.output_dir)

    base.BASE = Path(".")
    base.TABLES = tables_dir
    base.OUT = output_dir / "tables"
    base.OUT_TEXT = output_dir
    base.main()
    print(f"booster_v2_done: tables={tables_dir} output={output_dir}")


if __name__ == "__main__":
    main()
