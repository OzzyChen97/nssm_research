"""MMLongBench loader adapter for NSSM.

This adapter intentionally keeps preprocessing simple and transparent:
- It reads original MMLongBench json/jsonl files directly from `mmlb_data`.
- It resolves image paths against `mmlb_image`.
- It converts heterogeneous task fields into a unified sample schema consumed by
  the NSSM inference engine.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class MMLongBenchSample:
    """Unified sample format for NSSM experiments."""

    sample_id: str
    prompt: str
    answer: Any
    image_paths: List[str]
    raw: Dict[str, Any] = field(default_factory=dict)


def _is_image_like(path: str) -> bool:
    lower = path.lower()
    return lower.endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp"))


def _read_json_or_jsonl(file_path: Path) -> Iterable[Dict[str, Any]]:
    suffix = file_path.suffix.lower()
    with file_path.open("r", encoding="utf-8") as f:
        if suffix == ".jsonl":
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
            return

        loaded = json.load(f)
        if isinstance(loaded, list):
            for item in loaded:
                yield item
            return
        if isinstance(loaded, dict):
            # Some exports use {"data": [...]}.
            if "data" in loaded and isinstance(loaded["data"], list):
                for item in loaded["data"]:
                    yield item
                return
            # Otherwise treat as single sample.
            yield loaded
            return

        raise ValueError(f"Unsupported JSON structure in file: {file_path}")


def _build_context_text(record: Dict[str, Any]) -> str:
    if isinstance(record.get("context"), str) and record["context"].strip():
        return record["context"]

    ctxs = record.get("ctxs")
    if isinstance(ctxs, list) and ctxs:
        if isinstance(ctxs[0], dict):
            chunks = []
            for item in ctxs:
                title = item.get("title", "")
                text = item.get("text", "")
                if title and text:
                    chunks.append(f"Document (Title: {title}): {text}")
                elif text:
                    chunks.append(text)
            return "\n\n".join(chunks)
        if isinstance(ctxs[0], str):
            # Image-only contexts from visual haystack style tasks.
            return "\n".join("<image>" for _ in ctxs)
    return ""


def _resolve_image_paths(record: Dict[str, Any], image_root: Path) -> List[str]:
    candidates: List[str] = []
    if isinstance(record.get("image"), str):
        candidates.append(record["image"])
    if isinstance(record.get("image_list"), list):
        candidates.extend([x for x in record["image_list"] if isinstance(x, str)])
    if isinstance(record.get("page_list"), list):
        candidates.extend([x for x in record["page_list"] if isinstance(x, str)])

    ctxs = record.get("ctxs")
    if isinstance(ctxs, list) and ctxs and isinstance(ctxs[0], str):
        # For visual-haystack like samples, ctxs are image paths.
        if all(_is_image_like(x) for x in ctxs):
            candidates.extend(ctxs)

    resolved: List[str] = []
    for rel in candidates:
        if os.path.isabs(rel):
            resolved.append(rel)
        else:
            resolved.append(str(image_root / rel))
    return resolved


def _build_prompt(record: Dict[str, Any]) -> str:
    question = record.get("question", "")
    context_text = _build_context_text(record)
    if context_text:
        return f"{context_text}\n\nQuestion: {question}"
    return str(question)


def load_mmlongbench_samples(
    data_root: str,
    image_root: str,
    dataset_file: str,
    max_samples: Optional[int] = None,
) -> List[MMLongBenchSample]:
    """Load MMLongBench samples into a unified NSSM format.

    Args:
        data_root: Root directory of MMLongBench text files (`mmlb_data`).
        image_root: Root directory of corresponding images (`mmlb_image`).
        dataset_file: Relative path under `data_root`, e.g.
            ``documentQA/mmlongdoc_K128.jsonl``.
        max_samples: Optional cap for fast debugging.
    """

    data_path = Path(data_root) / dataset_file
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    image_root_path = Path(image_root)
    samples: List[MMLongBenchSample] = []

    for idx, record in enumerate(_read_json_or_jsonl(data_path)):
        sample_id = str(record.get("id", f"sample_{idx}"))
        sample = MMLongBenchSample(
            sample_id=sample_id,
            prompt=_build_prompt(record),
            answer=record.get("answer"),
            image_paths=_resolve_image_paths(record, image_root_path),
            raw=record,
        )
        samples.append(sample)
        if max_samples is not None and len(samples) >= max_samples:
            break

    return samples

