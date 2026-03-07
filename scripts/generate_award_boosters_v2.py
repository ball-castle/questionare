#!/usr/bin/env python3
"""本脚本用于以可配置路径运行国奖补强产物生成流程。"""

from __future__ import annotations

import argparse
from pathlib import Path


def _load_base_module():
    import generate_award_boosters as m  # type: ignore

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
