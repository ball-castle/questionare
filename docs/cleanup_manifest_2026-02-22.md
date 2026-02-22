# Cleanup Manifest (2026-02-22)

## Scope
Implement medium cleanup for questionnaire processing (880口径), with staged isolation before final deletion.

## Required Paths Kept
- scripts/run_current_pipeline.py
- scripts/run_questionnaire_analysis_v2.py
- scripts/convert_961_to_108.py
- scripts/generate_award_boosters_v2.py
- scripts/fill_pending_doc_v2.py
- scripts/check_pending_doc_consistency_v2.py
- scripts/qp_io.py
- scripts/qp_stats.py
- archive_legacy/scripts/run_questionnaire_analysis.py
- archive_legacy/scripts/generate_award_boosters.py
- data/raw/数据.xlsx
- docs/待填数据.docx
- output_current/**
- README.md
- pyproject.toml
- uv.lock
- .python-version
- .gitignore
- example/**
- Root reference docx files

## Isolated To
- archive_legacy/_to_delete_20260222/

## Isolated Items
- archive_legacy/output/
- archive_legacy/output_runs/
- archive_legacy/output_compare/
- archive_legacy/scripts/check_pending_doc_consistency.py
- archive_legacy/scripts/compare_dataset_runs.py
- archive_legacy/scripts/fill_pending_doc.py
- archive_legacy/scripts/run_dual_pipeline.py
- archive_legacy/scripts/__pycache__/

## Temp Cleanup
- docs/~WRL*.tmp removed if present.

## Verification Plan
1. Run current pipeline with data/raw/数据.xlsx and docs/待填数据.docx.
2. Verify raw=961 and remain_n_revised=880.
3. Verify pending-doc triplet outputs and 六七相关 artifacts.
4. Check no reverse dependency on isolated legacy output paths and isolated 4 scripts.
5. If all pass, remove archive_legacy/_to_delete_20260222/.
