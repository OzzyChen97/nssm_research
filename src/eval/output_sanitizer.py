"""Prompt-aware output cleanup for official MMLongBench formatting."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional


ANSWER_PREFIX_RE = re.compile(r"^(?:answer|assistant|label)\s*:\s*", flags=re.IGNORECASE)
YES_NO_RE = re.compile(r"\b(?:yes|no)\b", flags=re.IGNORECASE)
CHOICE_RE = re.compile(r"\b([A-Z])\b")
JSON_LIST_RE = re.compile(r"\[[^\[\]]*\]")


@dataclass(frozen=True)
class SanitizedAnswer:
    answer: str
    mode: str = "none"


def _strip_known_prefixes(text: str) -> str:
    cleaned = text.strip()
    while True:
        updated = ANSWER_PREFIX_RE.sub("", cleaned, count=1).strip()
        if updated == cleaned:
            return cleaned
        cleaned = updated


def _extract_label(text: str) -> Optional[str]:
    cleaned = _strip_known_prefixes(text)
    if not cleaned:
        return None
    first_line = cleaned.splitlines()[0].strip()
    if not first_line:
        return None
    if ":" in first_line:
        first_line = first_line.split(":", 1)[-1].strip()
    token = first_line.split()[0].strip().strip(",.;:()[]{}")
    return token or None


def _extract_last_yes_no(text: str) -> Optional[str]:
    matches = YES_NO_RE.findall(text)
    if not matches:
        return None
    return matches[-1].capitalize()


def _extract_last_choice_letter(text: str) -> Optional[str]:
    cleaned = _strip_known_prefixes(text)
    matches = CHOICE_RE.findall(cleaned)
    if not matches:
        return None
    return matches[-1]


def _extract_last_json_list(text: str) -> Optional[str]:
    matches = JSON_LIST_RE.findall(text)
    for candidate in reversed(matches):
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, list):
            return json.dumps(value)
    return None


def sanitize_answer(prompt: str, answer: str) -> SanitizedAnswer:
    cleaned = answer.strip()
    if not cleaned:
        return SanitizedAnswer(answer="")

    if "Only output \"label:" in prompt or "Now classify this image:" in prompt:
        label = _extract_label(cleaned)
        if label:
            return SanitizedAnswer(answer=label, mode="label")

    if "Please answer the question in Yes or No" in prompt:
        yes_no = _extract_last_yes_no(cleaned)
        if yes_no is not None:
            return SanitizedAnswer(answer=yes_no, mode="yes_no")

    if "option's letter (A, B, etc.)" in prompt:
        choice = _extract_last_choice_letter(cleaned)
        if choice is not None:
            return SanitizedAnswer(answer=choice, mode="choice_letter")

    if "Only output the results in JSON format" in prompt:
        json_list = _extract_last_json_list(cleaned)
        if json_list is not None:
            return SanitizedAnswer(answer=json_list, mode="json_list")

    stripped = _strip_known_prefixes(cleaned)
    return SanitizedAnswer(answer=stripped, mode="prefix_strip" if stripped != cleaned else "none")
