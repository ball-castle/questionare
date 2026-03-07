from __future__ import annotations

import argparse
import runpy
import sys


COMMANDS: dict[str, tuple[str, str]] = {
    "pipeline": ("questionnaire_analysis.run_current_pipeline", "Run the main analysis pipeline."),
    "rescreen-880": ("questionnaire_analysis.run_rescreen_to_data", "Persist the 961 -> 880 rescreen outputs."),
    "reliability-880": ("questionnaire_analysis.run_reliability_validity_880", "Run reliability and validity analysis on the 880 sample."),
    "logit": ("questionnaire_analysis.run_logit", "Run the Logit modeling workflow."),
    "clustering": ("questionnaire_analysis.run_clustering", "Run the clustering workflow."),
    "sem": ("questionnaire_analysis.run_sem", "Run the SEM workflow."),
    "demo": ("questionnaire_analysis.demo", "Write a demo output bundle without real survey data."),
}


def build_parser() -> argparse.ArgumentParser:
    lines = ["available commands:"]
    for name, (_, desc) in COMMANDS.items():
        lines.append(f"  {name:<15} {desc}")
    parser = argparse.ArgumentParser(
        prog="questionnaire-analysis",
        description="Questionnaire analysis unified CLI.",
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

    module_name, _ = COMMANDS[ns.command]
    old_argv = sys.argv[:]
    try:
        sys.argv = [module_name, *ns.args]
        runpy.run_module(module_name, run_name="__main__")
    finally:
        sys.argv = old_argv
