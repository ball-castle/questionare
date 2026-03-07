from __future__ import annotations

import sys
from pathlib import Path


def _load_main():
    try:
        from questionnaire_analysis.run_clustering import main
    except ModuleNotFoundError:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from questionnaire_analysis.run_clustering import main
    return main


if __name__ == "__main__":
    _load_main()()

