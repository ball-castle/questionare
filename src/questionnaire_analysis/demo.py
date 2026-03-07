from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def write_demo_bundle(output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).isoformat()

    survey_header = ["respondent_id", "C001", "C002", "C003", "C088", "C090", "C091"]
    survey_rows = [
        [1, 1, 3, 4, 4, 4, 4],
        [2, 2, 2, 3, 5, 5, 4],
        [3, 1, 4, 5, 3, 3, 3],
        [4, 2, 1, 2, 2, 2, 2],
    ]
    _write_csv(tables_dir / "survey_clean.csv", survey_header, survey_rows)

    _write_csv(
        tables_dir / "信度分析表.csv",
        ["block", "alpha", "n_complete"],
        [
            ["感知维度(52-63,65)", "0.9123", 880],
            ["重要度维度(66-75)", "0.8871", 880],
            ["表现维度(76-85)", "0.9015", 880],
            ["认知维度(86-89)", "0.8458", 880],
            ["综合量表(52-63,65,66-85,86-89)", "0.9432", 880],
        ],
    )
    _write_csv(
        tables_dir / "效度分析表.csv",
        ["n_complete", "kmo", "bartlett_chi2", "bartlett_df", "bartlett_p"],
        [[880, "0.9234", "12543.827", 666, "0.0"]],
    )
    _write_csv(
        tables_dir / "建议落地行动矩阵.csv",
        ["priority", "theme", "action", "owner_hint"],
        [
            ["P1", "文化体验", "扩充非遗体验项目排期并缩短排队时间", "运营"],
            ["P1", "服务专业度", "为一线人员补充中医药知识问答卡片", "培训"],
            ["P2", "传播转化", "联动短视频与门店活动做同主题传播", "市场"],
        ],
    )

    summary = "\n".join(
        [
            "Questionnaire Analysis Demo Output",
            "",
            "This bundle is generated from synthetic records so that contributors can inspect the output structure",
            "without using the private survey workbook.",
            "",
            "Included artifacts:",
            "- tables/survey_clean.csv",
            "- tables/信度分析表.csv",
            "- tables/效度分析表.csv",
            "- tables/建议落地行动矩阵.csv",
            "- run_metadata.json",
            "- pipeline_manifest.json",
        ]
    )
    (output_dir / "问卷数据处理与分析_结果摘要.txt").write_text(summary, encoding="utf-8")

    metadata = {
        "generated_at_utc": timestamp,
        "source_pipeline": "demo_bundle",
        "input_format": "synthetic_demo",
        "n_samples": len(survey_rows),
        "remain_n_revised": len(survey_rows),
        "invalid_n_revised": 0,
        "notes": "Synthetic demo output for repository verification only.",
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "pipeline": "demo_bundle",
        "started_at": timestamp,
        "finished_at": timestamp,
        "status": "ok",
        "input_xlsx": None,
        "input_format": "synthetic_demo",
        "output_dir": str(output_dir),
        "steps": [
            {"step": "demo", "status": "ok"},
        ],
        "key_outputs": [
            str(output_dir / "问卷数据处理与分析_结果摘要.txt"),
            str(tables_dir / "survey_clean.csv"),
            str(tables_dir / "信度分析表.csv"),
            str(tables_dir / "效度分析表.csv"),
            str(tables_dir / "建议落地行动矩阵.csv"),
            str(output_dir / "run_metadata.json"),
            str(output_dir / "pipeline_manifest.json"),
        ],
    }
    (output_dir / "pipeline_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a synthetic demo output bundle.")
    parser.add_argument("--output-dir", default="demo-output", help="Directory for generated demo artifacts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = write_demo_bundle(Path(args.output_dir))
    print(f"demo_bundle_written: out={out_dir}")


if __name__ == "__main__":
    main()
