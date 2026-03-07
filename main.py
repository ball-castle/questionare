from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

COMMANDS: dict[str, tuple[Path, str]] = {
    "pipeline": (ROOT / "scripts" / "run_current_pipeline.py", "运行主分析流程"),
    "rescreen-880": (ROOT / "scripts" / "run_rescreen_to_data.py", "执行 961 -> 880 重筛"),
    "reliability-880": (ROOT / "scripts" / "run_reliability_validity_880.py", "运行 880 样本信效度分析"),
    "logit": (ROOT / "scripts" / "run_logit.py", "运行 Logit 建模"),
    "clustering": (ROOT / "scripts" / "run_clustering.py", "运行聚类分析"),
    "sem": (ROOT / "scripts" / "run_sem.py", "运行结构方程模型分析"),
}


def build_parser() -> argparse.ArgumentParser:
    lines = ["available commands:"]
    for name, (_, desc) in COMMANDS.items():
        lines.append(f"  {name:<15} {desc}")
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="问卷分析统一 CLI 入口。",
        epilog="\n".join(lines),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
    )
    parser.add_argument("command", nargs="?")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    return parser


def main() -> None:
    parser = build_parser()
    argv = sys.argv[1:]

    if not argv or argv[0] in {"-h", "--help"}:
        parser.print_help()
        return

    ns = parser.parse_args(argv)

    if ns.command is None:
        parser.print_help()
        return

    if ns.command not in COMMANDS:
        parser.error(f"unknown command: {ns.command}")

    script_path, _ = COMMANDS[ns.command]
    subprocess.run([sys.executable, str(script_path), *ns.args], check=True)


if __name__ == "__main__":
    main()
