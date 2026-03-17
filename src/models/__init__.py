"""Model interfaces and factories for NSSM.

This module exposes a small backend protocol that keeps the NSSM pipeline
model-agnostic. The first backend implementation is Qwen2.5-VL, but the
pipeline is intentionally written against this interface so other VLMs can be
plugged in with minimal adaptation code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch


@dataclass
class BackendConfig:
    """Runtime configuration shared by all backend adapters."""

    model_name: str
    model_local_path: Optional[str] = None
    precision: str = "bfloat16"
    device_map: str = "auto"
    use_flash_attention_2: bool = True
    torch_compile: bool = False
    trust_remote_code: bool = True
    image_resize: Optional[float] = None
    max_image_num: Optional[int] = None
    force_textual_fallback: bool = False


class BaseVLMBackend(ABC):
    """Backend protocol consumed by NSSM.

    Implementations are expected to expose three capabilities:
    1. Perception: extract dense visual tokens from raw media.
    2. Query understanding: encode text prompts into token embeddings.
    3. Reasoning: generate final responses, optionally conditioned on routed
       NSSM slot embeddings.
    """

    @property
    @abstractmethod
    def hidden_size(self) -> int:
        """Return the language hidden size used for slot injection."""

    @abstractmethod
    def extract_visual_tokens(self, media_inputs: Any, prompt: str) -> torch.Tensor:
        """Extract visual tokens with shape ``[B, L_v, D]``."""

    @abstractmethod
    def encode_prompt(self, prompt: str) -> torch.Tensor:
        """Encode prompt text into token embeddings with shape ``[B, L_t, D]``."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        media_inputs: Any = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> str:
        """Run standard model generation without explicit NSSM slot injection."""

    @abstractmethod
    def generate_with_selected_slots(
        self,
        prompt: str,
        selected_slots: torch.Tensor,
        slot_names: Optional[Sequence[str]] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> str:
        """Run generation conditioned on routed NSSM slots."""


def build_backend(backend_name: str, config: BackendConfig) -> BaseVLMBackend:
    """Factory for backend adapters.

    Args:
        backend_name: Short backend id, e.g., ``"qwen"``.
        config: Shared backend runtime config.

    Returns:
        Concrete backend adapter implementing :class:`BaseVLMBackend`.
    """

    normalized = backend_name.strip().lower()
    if normalized == "qwen":
        from .qwen_nssm_wrapper import QwenNSSMWrapper

        return QwenNSSMWrapper(config)

    raise ValueError(
        f"Unsupported backend '{backend_name}'. "
        "Implement a BaseVLMBackend adapter and register it in build_backend()."
    )
