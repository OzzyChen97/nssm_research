"""Explicit naming module for NSSM dynamic slots.

The purpose of this stage is to transform latent slot vectors into explicit,
human-readable metadata. This supports:
1. Interpretable memory traces in long-context VLM inference.
2. Name-based routing between user query intent and candidate slots.

Default policy is intentionally lightweight:
- Prototype naming from prompt anchors + slot salience scores.
- Optional LLM refinement can be enabled for higher linguistic quality.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import torch
from torch import Tensor

from src.models import BaseVLMBackend


_STOPWORDS = {
    "a",
    "an",
    "the",
    "to",
    "of",
    "in",
    "on",
    "for",
    "is",
    "are",
    "was",
    "were",
    "and",
    "or",
    "with",
    "what",
    "which",
    "who",
    "when",
    "where",
    "how",
    "why",
    "please",
}


@dataclass
class SlotMetadata:
    """Named representation of a dynamic slot."""

    slot_id: int
    name: str
    confidence: float
    auxiliary: dict = field(default_factory=dict)


class PrototypeSlotNamer:
    """Generate short textual labels for dynamic slots."""

    def __init__(self, max_label_words: int = 6) -> None:
        self.max_label_words = max_label_words

    def _extract_prompt_keywords(self, prompt: str, k: int = 8) -> List[str]:
        tokens = re.findall(r"[A-Za-z0-9]+", prompt.lower())
        filtered = [tok for tok in tokens if tok not in _STOPWORDS and len(tok) > 2]
        if not filtered:
            return ["visual", "evidence"]
        seen = set()
        deduped: List[str] = []
        for tok in filtered:
            if tok in seen:
                continue
            deduped.append(tok)
            seen.add(tok)
            if len(deduped) >= k:
                break
        return deduped

    def _prototype_label(self, slot_id: int, keyword: str) -> str:
        label = f"{keyword} evidence slot {slot_id}"
        words = label.split()
        if len(words) <= self.max_label_words:
            return label
        return " ".join(words[: self.max_label_words])

    def _refine_with_backend(
        self,
        backend: BaseVLMBackend,
        prompt: str,
        metadata: List[SlotMetadata],
        group_size: int,
    ) -> List[SlotMetadata]:
        refined = list(metadata)
        for start in range(0, len(metadata), group_size):
            chunk = metadata[start : start + group_size]
            numbered = "\n".join(
                f"{item.slot_id}: {item.name} (conf={item.confidence:.3f})"
                for item in chunk
            )
            rewrite_prompt = (
                "Rewrite each slot label into at most six words.\n"
                "Preserve semantics and return one line per slot in format "
                "'slot_id: short label'.\n\n"
                f"User question: {prompt}\n\nSlots:\n{numbered}"
            )
            text = backend.generate(
                prompt=rewrite_prompt,
                media_inputs=None,
                max_new_tokens=80,
                temperature=0.0,
                top_p=1.0,
            )
            parsed = {}
            for line in text.splitlines():
                if ":" not in line:
                    continue
                left, right = line.split(":", 1)
                left = left.strip()
                right = right.strip()
                if not left.isdigit() or not right:
                    continue
                parsed[int(left)] = " ".join(right.split()[: self.max_label_words])
            for item in chunk:
                if item.slot_id in parsed:
                    item.name = parsed[item.slot_id]
        return refined

    @torch.no_grad()
    def name_slots(
        self,
        dynamic_slots: Tensor,
        prompt_embeds: Tensor,
        prompt_text: str,
        backend: Optional[BaseVLMBackend] = None,
        enable_llm_refine: bool = False,
        llm_group_size: int = 8,
    ) -> List[SlotMetadata]:
        """Assign explicit names to slots.

        Args:
            dynamic_slots: Slot embeddings ``[B, S, D]``.
            prompt_embeds: Prompt token embeddings ``[B, L_t, D]``.
            prompt_text: Raw user query string.
            backend: Optional backend for LLM-based label rewrite.
            enable_llm_refine: Whether to run optional label rewriting.
            llm_group_size: Number of slots per rewrite call.
        """

        if dynamic_slots.ndim != 3 or dynamic_slots.size(0) != 1:
            raise ValueError(
                "PrototypeSlotNamer currently expects dynamic_slots with shape [1, S, D]."
            )
        if prompt_embeds.ndim != 3 or prompt_embeds.size(0) != 1:
            raise ValueError(
                "PrototypeSlotNamer currently expects prompt_embeds with shape [1, L, D]."
            )

        slots = dynamic_slots[0]
        prompt_vec = torch.nn.functional.normalize(
            prompt_embeds[0].mean(dim=0, keepdim=True), dim=-1
        )
        slot_vecs = torch.nn.functional.normalize(slots, dim=-1)
        salience = (slot_vecs @ prompt_vec.transpose(0, 1)).squeeze(-1)
        confidences = torch.softmax(salience, dim=0)

        keywords = self._extract_prompt_keywords(prompt_text)
        metadata: List[SlotMetadata] = []
        for slot_id in range(slots.size(0)):
            keyword = keywords[slot_id % len(keywords)]
            metadata.append(
                SlotMetadata(
                    slot_id=slot_id,
                    name=self._prototype_label(slot_id=slot_id, keyword=keyword),
                    confidence=float(confidences[slot_id].item()),
                    auxiliary={"prototype_score": float(salience[slot_id].item())},
                )
            )

        if enable_llm_refine and backend is not None:
            metadata = self._refine_with_backend(
                backend=backend,
                prompt=prompt_text,
                metadata=metadata,
                group_size=max(1, llm_group_size),
            )

        return metadata

