#!/usr/bin/env python3
"""Outline-driven rendering helpers for chapter 6/7 reports."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

EVIDENCE_TAG_RE = re.compile(r"\[证据：([^\]]+)\]")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
IMAGE_RE = re.compile(r"^!\[(.*?)\]\((.*?)\)\s*$")
BULLET_RE = re.compile(r"^\s*-\s+(.*)$")


def _fmt_path(path: str | Path) -> str:
    return Path(path).as_posix()


def _looks_like_evidence_path(path_text: str) -> bool:
    if "/" in path_text or "\\" in path_text:
        return True
    lowered = path_text.lower()
    return lowered.endswith((".csv", ".txt", ".png", ".docx", ".md", ".json", ".xlsx"))


def normalize_outline_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n")
    normalized = normalized.replace("output_current/", "output/")
    normalized = normalized.replace("output_current\\", "output\\")
    return normalized


def render_outline_markdown(
    outline_text: str,
    outline_path: Path,
    missing_items: Iterable[str] | None = None,
) -> str:
    lines: list[str] = [f"> 大纲来源：`{_fmt_path(outline_path)}`"]
    for path in list(missing_items or []):
        lines.append(f"> 输入缺失：`{_fmt_path(path)}`")
    lines.append("")
    lines.append(outline_text.rstrip())
    lines.append("")
    return "\n".join(lines)


def parse_markdown_blocks(markdown_text: str) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    para_lines: list[str] = []

    def flush_para() -> None:
        if not para_lines:
            return
        blocks.append({"type": "paragraph", "text": "\n".join(para_lines).strip()})
        para_lines.clear()

    for raw_line in markdown_text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            flush_para()
            continue

        heading_match = HEADING_RE.match(stripped)
        if heading_match:
            flush_para()
            level = len(heading_match.group(1))
            text = heading_match.group(2).strip()
            blocks.append({"type": "heading", "level": level, "text": text})
            continue

        image_match = IMAGE_RE.match(stripped)
        if image_match:
            flush_para()
            caption = image_match.group(1).strip()
            img_path = image_match.group(2).strip()
            path_obj = Path(img_path)
            blocks.append(
                {
                    "type": "image",
                    "path": _fmt_path(path_obj),
                    "caption": caption,
                    "exists": path_obj.exists(),
                }
            )
            continue

        bullet_match = BULLET_RE.match(line)
        if bullet_match:
            flush_para()
            blocks.append({"type": "bullet", "text": bullet_match.group(1).strip()})
            continue

        if stripped == "---":
            flush_para()
            continue

        if stripped.startswith("> "):
            flush_para()
            blocks.append({"type": "paragraph", "text": stripped[2:].strip()})
            continue

        para_lines.append(stripped)

    flush_para()
    return blocks


def extract_evidence_rows(markdown_text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    heading_stack: dict[int, str] = {}
    last_statement = ""

    for raw_line in markdown_text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue

        heading_match = HEADING_RE.match(stripped)
        if heading_match:
            level = len(heading_match.group(1))
            heading_text = heading_match.group(2).strip()
            heading_stack[level] = heading_text
            for old_level in [k for k in heading_stack if k > level]:
                del heading_stack[old_level]
            continue

        matches = EVIDENCE_TAG_RE.findall(stripped)
        if not matches:
            candidate = re.sub(r"^\s*-\s*", "", stripped).strip()
            if candidate:
                last_statement = candidate
            continue

        section_parts = [heading_stack[k] for k in sorted(heading_stack)]
        section = " > ".join(section_parts) if section_parts else "未分节"

        statement = EVIDENCE_TAG_RE.sub("", stripped)
        statement = re.sub(r"^\s*-\s*", "", statement).strip()
        if not statement:
            statement = last_statement or "（证据引用）"
        else:
            last_statement = statement

        for group in matches:
            for token in re.split(r"[，,、]", group):
                path_text = token.strip().strip("`").strip()
                if not path_text:
                    continue
                path_text = path_text.replace("output_current/", "output/")
                path_text = path_text.replace("output_current\\", "output\\")
                if not _looks_like_evidence_path(path_text):
                    continue
                evidence_path = _fmt_path(path_text)
                key = (section, statement, evidence_path)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        "section": section,
                        "statement": statement,
                        "evidence_path": evidence_path,
                    }
                )

    return rows


def build_outline_report(
    outline_md: Path,
    title: str,
    missing_items: Iterable[str] | None = None,
) -> tuple[dict[str, Any], str]:
    outline_md = Path(outline_md)
    if not outline_md.exists():
        raise FileNotFoundError(f"missing required outline: {outline_md}")

    raw_text = outline_md.read_text(encoding="utf-8")
    normalized_text = normalize_outline_text(raw_text)
    markdown_text = render_outline_markdown(
        outline_text=normalized_text,
        outline_path=outline_md,
        missing_items=missing_items,
    )

    warnings = [f"missing: {_fmt_path(path)}" for path in list(missing_items or [])]
    content = {
        "title": title,
        "blocks": parse_markdown_blocks(markdown_text),
        "evidence_rows": extract_evidence_rows(markdown_text),
        "warnings": warnings,
        "outline_path": _fmt_path(outline_md),
        "outline_exists": True,
    }
    return content, markdown_text
