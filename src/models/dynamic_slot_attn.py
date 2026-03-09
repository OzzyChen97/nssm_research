"""Dynamic prompt-aware slot construction for NSSM.

This module implements the core methodological claim of NSSM:
slots are *not* static memory vectors obtained offline. Instead, at every
inference step we construct slots on-the-fly by conditioning visual aggregation
on the *current* user query. The module is intentionally lightweight so that:

1. It can be used zero-shot on top of pretrained VLM features.
2. It can be lightly tuned on long-context supervision (e.g., LoRA).
3. It remains practical for 100k+ visual token regimes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import Tensor, nn


@dataclass
class SlotAttentionStats:
    """Optional debug container returned by the aggregator."""

    prompt_condition_norm: float
    slot_query_norm: float
    slot_output_norm: float


class QueryAwareSlotAggregator(nn.Module):
    """Inference-time query-aware slot constructor.

    Given a long sequence of visual tokens ``visual_tokens`` and the embedding
    sequence of the current prompt ``text_prompt_embeds``, this module creates a
    fixed-size set of semantic slots:

    ``dynamic_slots = Agg(visual_tokens, text_prompt_embeds)``

    Design choices:
    - Uses learnable slot query anchors to preserve stable slot capacity.
    - Uses prompt-conditioned query shifting so slots become query-aware at
      inference time.
    - Uses cross-attention from conditioned slot queries to all visual tokens.
    - Keeps parameter count small for long-context practical deployment.
    """

    def __init__(
        self,
        hidden_size: int,
        num_slots: int = 256,
        num_heads: int = 8,
        dropout: float = 0.0,
        prompt_mlp_ratio: float = 2.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"num_heads ({num_heads})."
            )

        self.hidden_size = hidden_size
        self.num_slots = num_slots

        self.slot_queries = nn.Parameter(torch.empty(num_slots, hidden_size))
        nn.init.normal_(self.slot_queries, mean=0.0, std=0.02)

        mlp_hidden = int(hidden_size * prompt_mlp_ratio)
        self.prompt_adapter = nn.Sequential(
            nn.LayerNorm(hidden_size, eps=eps),
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_size),
        )
        self.prompt_gate = nn.Linear(hidden_size, hidden_size)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.pre_ffn_norm = nn.LayerNorm(hidden_size, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.final_norm = nn.LayerNorm(hidden_size, eps=eps)

    def _masked_mean(self, x: Tensor, mask: Optional[Tensor]) -> Tensor:
        """Compute masked mean over sequence length."""

        if mask is None:
            return x.mean(dim=1)
        if mask.ndim != 2:
            raise ValueError(f"prompt_mask must have shape [B, L], got {mask.shape}")
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (x * mask.unsqueeze(-1).to(x.dtype)).sum(dim=1) / denom

    def forward(
        self,
        visual_tokens: Tensor,
        text_prompt_embeds: Tensor,
        visual_mask: Optional[Tensor] = None,
        prompt_mask: Optional[Tensor] = None,
        return_debug: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, SlotAttentionStats]]:
        """Construct dynamic slots for the current prompt.

        Args:
            visual_tokens: Visual sequence ``[B, L_v, D]``.
            text_prompt_embeds: Prompt embeddings ``[B, L_t, D]``.
            visual_mask: Optional valid-token mask for visual tokens ``[B, L_v]``
                where 1 means valid and 0 means padding.
            prompt_mask: Optional valid-token mask for prompt tokens ``[B, L_t]``.
            return_debug: Whether to return summary statistics.

        Returns:
            Either dynamic slots ``[B, S, D]`` or a tuple
            ``(dynamic_slots, SlotAttentionStats)``.
        """

        if visual_tokens.ndim != 3:
            raise ValueError(
                f"visual_tokens must be rank-3 [B, L_v, D], got {visual_tokens.shape}"
            )
        if text_prompt_embeds.ndim != 3:
            raise ValueError(
                "text_prompt_embeds must be rank-3 [B, L_t, D], "
                f"got {text_prompt_embeds.shape}"
            )
        if visual_tokens.size(0) != text_prompt_embeds.size(0):
            raise ValueError("visual and prompt batch sizes must match.")
        if visual_tokens.size(-1) != self.hidden_size:
            raise ValueError(
                f"visual token hidden size mismatch: expected {self.hidden_size}, "
                f"got {visual_tokens.size(-1)}"
            )
        if text_prompt_embeds.size(-1) != self.hidden_size:
            raise ValueError(
                f"prompt hidden size mismatch: expected {self.hidden_size}, "
                f"got {text_prompt_embeds.size(-1)}"
            )

        batch_size = visual_tokens.size(0)

        prompt_summary = self._masked_mean(text_prompt_embeds, prompt_mask)
        prompt_condition = self.prompt_adapter(prompt_summary)
        prompt_gate = torch.sigmoid(self.prompt_gate(prompt_condition)).unsqueeze(1)

        base_queries = self.slot_queries.unsqueeze(0).expand(batch_size, -1, -1)
        conditioned_queries = base_queries + prompt_gate * prompt_condition.unsqueeze(1)

        key_padding_mask = None
        if visual_mask is not None:
            if visual_mask.ndim != 2:
                raise ValueError(
                    f"visual_mask must have shape [B, L_v], got {visual_mask.shape}"
                )
            key_padding_mask = ~visual_mask.to(torch.bool)

        slot_updates, _ = self.cross_attn(
            query=conditioned_queries,
            key=visual_tokens,
            value=visual_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        slots = conditioned_queries + slot_updates
        slots = slots + self.ffn(self.pre_ffn_norm(slots))
        slots = self.final_norm(slots)

        if not return_debug:
            return slots

        stats = SlotAttentionStats(
            prompt_condition_norm=float(prompt_condition.norm(dim=-1).mean().item()),
            slot_query_norm=float(conditioned_queries.norm(dim=-1).mean().item()),
            slot_output_norm=float(slots.norm(dim=-1).mean().item()),
        )
        return slots, stats

