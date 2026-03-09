"""Name-aware memory routing for NSSM.

Given named dynamic slots and a user query, this module chooses the top-K slots
that should enter the final System-2 reasoning stage. Routing combines:
1. Text-to-name relevance (query vs. slot textual labels).
2. Query-to-slot embedding relevance (query vs. slot vectors).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
from torch import Tensor

from src.models import BaseVLMBackend
from src.pipeline.slot_namer import SlotMetadata


@dataclass
class RouterOutput:
    """Router decision for Step 4 of NSSM."""

    indices: List[int]
    scores: List[float]
    selected_slots: Tensor
    selected_metadata: List[SlotMetadata]


class NameAwareMemoryRouter:
    """Fuse textual and embedding signals to select top-K slots."""

    def __init__(self, top_k: int = 32, alpha_text_name: float = 0.6) -> None:
        if not (0.0 <= alpha_text_name <= 1.0):
            raise ValueError("alpha_text_name must be in [0, 1].")
        self.top_k = top_k
        self.alpha_text_name = alpha_text_name

    @torch.no_grad()
    def route(
        self,
        prompt_text: str,
        prompt_embeds: Tensor,
        dynamic_slots: Tensor,
        slot_metadata: Sequence[SlotMetadata],
        backend: Optional[BaseVLMBackend] = None,
    ) -> RouterOutput:
        """Route to top-K slots.

        Args:
            prompt_text: Raw user prompt.
            prompt_embeds: Prompt embedding sequence ``[B, L_t, D]``.
            dynamic_slots: Slot tensor ``[B, S, D]``.
            slot_metadata: Slot names and metadata.
            backend: Optional model backend for text embedding.
        """

        if dynamic_slots.ndim != 3 or dynamic_slots.size(0) != 1:
            raise ValueError("dynamic_slots must have shape [1, S, D].")
        if prompt_embeds.ndim != 3 or prompt_embeds.size(0) != 1:
            raise ValueError("prompt_embeds must have shape [1, L, D].")

        slots = dynamic_slots[0]
        num_slots = slots.size(0)
        if num_slots == 0:
            return RouterOutput(
                indices=[],
                scores=[],
                selected_slots=dynamic_slots[:, :0, :],
                selected_metadata=[],
            )

        prompt_vec = torch.nn.functional.normalize(
            prompt_embeds[0].mean(dim=0), dim=-1
        )  # [D]
        slot_vecs = torch.nn.functional.normalize(slots, dim=-1)  # [S, D]
        slot_sim = slot_vecs @ prompt_vec  # [S]

        # Text-name relevance path.
        if backend is not None and slot_metadata:
            prompt_text_vec = torch.nn.functional.normalize(
                backend.encode_prompt(prompt_text)[0].mean(dim=0), dim=-1
            )
            name_vecs: List[Tensor] = []
            for item in slot_metadata:
                name_embed = backend.encode_prompt(item.name)[0].mean(dim=0)
                name_vecs.append(torch.nn.functional.normalize(name_embed, dim=-1))
            name_mat = torch.stack(name_vecs, dim=0)
            text_name_sim = name_mat @ prompt_text_vec
        else:
            # Confidence-only fallback when backend text embeddings are not available.
            if slot_metadata:
                text_name_sim = torch.tensor(
                    [float(item.confidence) for item in slot_metadata],
                    device=slots.device,
                    dtype=slots.dtype,
                )
            else:
                text_name_sim = torch.zeros(num_slots, device=slots.device, dtype=slots.dtype)

        fused = self.alpha_text_name * text_name_sim + (1.0 - self.alpha_text_name) * slot_sim
        k = min(self.top_k, num_slots)
        top = torch.topk(fused, k=k, dim=0)
        indices = top.indices.tolist()
        scores = top.values.tolist()
        selected_slots = dynamic_slots[:, top.indices, :]

        metadata_lookup = list(slot_metadata) if slot_metadata else []
        selected_metadata: List[SlotMetadata] = []
        for idx in indices:
            if 0 <= idx < len(metadata_lookup):
                selected_metadata.append(metadata_lookup[idx])
            else:
                selected_metadata.append(
                    SlotMetadata(
                        slot_id=idx,
                        name=f"slot_{idx}",
                        confidence=0.0,
                    )
                )

        return RouterOutput(
            indices=indices,
            scores=[float(x) for x in scores],
            selected_slots=selected_slots,
            selected_metadata=selected_metadata,
        )

