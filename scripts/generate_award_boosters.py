#!/usr/bin/env python3
"""统一国奖补强入口，支持自定义输入输出目录。"""

from __future__ import annotations

import argparse
from pathlib import Path

def run_boosters(tables_dir: Path, output_dir: Path) -> None:
    import award_booster_core as core

    tables_dir = Path(tables_dir)
    output_dir = Path(output_dir)

    core.BASE = Path(".")
    core.TABLES = tables_dir
    core.OUT = output_dir / "tables"
    core.OUT_TEXT = output_dir
    core.main()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate award booster artifacts.")
    parser.add_argument("--tables-dir", default="output/tables", help="Directory containing analysis tables csv files.")
    parser.add_argument("--output-dir", default="output", help="Run output root directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_boosters(Path(args.tables_dir), Path(args.output_dir))
    print(f"award_boosters_done: tables={args.tables_dir} output={args.output_dir}")


if __name__ == "__main__":
    main()
