"""Qwen2.5-VL backend adapter for NSSM.

This wrapper provides a stable backend interface for NSSM experiments:
1. Extract visual tokens from Qwen's vision tower.
2. Encode prompt text into embedding space for slot construction and routing.
3. Generate final answers with routed slot conditioning.

The implementation favors practical robustness:
- local-model path is attempted first for offline environments;
- generation with slot embeddings is attempted first;
- if backend generation APIs reject custom embeddings, a textual memory fallback
  is used to keep the experiment runnable.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from PIL import Image
from torch import Tensor
from transformers import AutoModelForImageTextToText, AutoProcessor

from . import BackendConfig, BaseVLMBackend

LOGGER = logging.getLogger(__name__)


def _precision_to_dtype(precision: str) -> torch.dtype:
    normalized = precision.strip().lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported precision '{precision}'.")


class QwenNSSMWrapper(BaseVLMBackend):
    """NSSM adapter around Qwen2.5-VL.

    Notes on slot injection:
    - Preferred path: inject selected slots as prefix embeddings through
      ``inputs_embeds`` and run generation.
    - Fallback path: convert slot metadata into explicit textual memory prompts.
    """

    def __init__(self, config: BackendConfig) -> None:
        self.config = config
        self.dtype = _precision_to_dtype(config.precision)
        self.image_resize = config.image_resize
        self.max_image_num = config.max_image_num
        self.force_textual_fallback = config.force_textual_fallback

        model_id = self._resolve_model_id(
            model_local_path=config.model_local_path,
            model_name=config.model_name,
        )
        self.model_id = model_id
        LOGGER.info("Loading Qwen backend from %s", model_id)

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=config.trust_remote_code,
            use_fast=True,
        )
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

        model_kwargs: Dict[str, Any] = {
            "torch_dtype": self.dtype,
            "device_map": config.device_map,
            "trust_remote_code": config.trust_remote_code,
        }
        if config.use_flash_attention_2:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                **model_kwargs,
            )
        except TypeError:
            # Some transformers versions may not accept attn_implementation for
            # specific checkpoints; this fallback keeps the code executable.
            model_kwargs.pop("attn_implementation", None)
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                **model_kwargs,
            )

        if config.torch_compile:
            try:
                self.model = torch.compile(self.model)
                LOGGER.info("Enabled torch.compile for Qwen backend.")
            except Exception as exc:  # pragma: no cover - runtime fallback
                LOGGER.warning("torch.compile failed for Qwen backend: %s", exc)

        self.model.eval()

    @property
    def hidden_size(self) -> int:
        """Return language hidden size for slot fusion."""

        if hasattr(self.model.config, "hidden_size"):
            return int(self.model.config.hidden_size)
        if hasattr(self.model.config, "text_config") and hasattr(
            self.model.config.text_config, "hidden_size"
        ):
            return int(self.model.config.text_config.hidden_size)
        raise AttributeError("Cannot determine model hidden_size from config.")

    def _resolve_model_id(self, model_local_path: Optional[str], model_name: str) -> str:
        if model_local_path:
            local = Path(model_local_path)
            if local.exists():
                return str(local)
            LOGGER.warning(
                "Configured local model path does not exist: %s. "
                "Falling back to model_name=%s",
                model_local_path,
                model_name,
            )
        return model_name

    def _model_device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _normalize_media_inputs(
        self, media_inputs: Any
    ) -> Tuple[List[Union[str, Image.Image]], List[Any]]:
        images: List[Union[str, Image.Image]] = []
        videos: List[Any] = []

        if media_inputs is None:
            return images, videos
        if isinstance(media_inputs, (str, Path, Image.Image)):
            return [media_inputs], videos
        if isinstance(media_inputs, (list, tuple)):
            return list(media_inputs), videos
        if isinstance(media_inputs, dict):
            images = list(media_inputs.get("images", []))
            videos = list(media_inputs.get("videos", []))
            if "image" in media_inputs:
                images.append(media_inputs["image"])
            if "video" in media_inputs:
                videos.append(media_inputs["video"])
            if self.max_image_num is not None and self.max_image_num >= 0:
                images = images[: self.max_image_num]
            return images, videos

        raise TypeError(
            "media_inputs must be None, image path/PIL, list/tuple, or dict with "
            "'images'/'videos' keys."
        )

    def _read_image(self, item: Union[str, Path, Image.Image]) -> Image.Image:
        if isinstance(item, Image.Image):
            image = item.convert("RGB")
        else:
            path = Path(item)
            if not path.exists():
                raise FileNotFoundError(f"Image file not found: {path}")
            image = Image.open(path).convert("RGB")
        if self.image_resize is not None and self.image_resize > 0 and self.image_resize != 1.0:
            width = max(1, int(round(image.width * self.image_resize)))
            height = max(1, int(round(image.height * self.image_resize)))
            image = image.resize((width, height), Image.LANCZOS)
        return image

    def _build_messages(
        self,
        prompt: str,
        images: Sequence[Image.Image],
        videos: Sequence[Any],
        add_generation_prompt: bool = True,
    ) -> Tuple[str, List[Image.Image], List[Any]]:
        content: List[Dict[str, Any]] = []
        for image in images:
            content.append({"type": "image", "image": image})
        for video in videos:
            content.append({"type": "video", "video": video})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        return text, list(images), list(videos)

    def _prepare_mm_inputs(
        self,
        prompt: str,
        media_inputs: Any = None,
        add_generation_prompt: bool = True,
    ) -> Dict[str, Tensor]:
        raw_images, raw_videos = self._normalize_media_inputs(media_inputs)
        images = [self._read_image(image) for image in raw_images]

        text, image_payload, video_payload = self._build_messages(
            prompt=prompt,
            images=images,
            videos=raw_videos,
            add_generation_prompt=add_generation_prompt,
        )
        inputs = self.processor(
            text=[text],
            images=image_payload if image_payload else None,
            videos=video_payload if video_payload else None,
            return_tensors="pt",
            padding=True,
        )
        device = self._model_device()
        return {k: v.to(device) for k, v in inputs.items()}

    @torch.no_grad()
    def extract_visual_tokens(self, media_inputs: Any, prompt: str) -> Tensor:
        """Extract dense visual token sequence from Qwen vision tower."""

        inputs = self._prepare_mm_inputs(
            prompt=prompt,
            media_inputs=media_inputs,
            add_generation_prompt=False,
        )
        visual_module = getattr(self.model, "visual", None)
        if visual_module is None:
            visual_module = getattr(getattr(self.model, "model", None), "visual", None)
        if visual_module is None:
            raise AttributeError(
                "Qwen model does not expose a `.visual` module needed for token extraction."
            )

        slot_chunks: List[Tensor] = []
        if "pixel_values" in inputs and inputs["pixel_values"] is not None:
            pixel_values = inputs["pixel_values"]
            image_grid = inputs.get("image_grid_thw", None)
            visual_dtype = getattr(visual_module, "dtype", self.dtype)
            image_embeds = visual_module(pixel_values.to(visual_dtype), grid_thw=image_grid)
            slot_chunks.append(image_embeds)

        if "pixel_values_videos" in inputs and inputs["pixel_values_videos"] is not None:
            pixel_values_videos = inputs["pixel_values_videos"]
            video_grid = inputs.get("video_grid_thw", None)
            visual_dtype = getattr(visual_module, "dtype", self.dtype)
            video_embeds = visual_module(
                pixel_values_videos.to(visual_dtype), grid_thw=video_grid
            )
            slot_chunks.append(video_embeds)

        if not slot_chunks:
            hidden = self.hidden_size
            return torch.zeros((1, 0, hidden), device=self._model_device(), dtype=self.dtype)

        all_visual = torch.cat(slot_chunks, dim=0).to(self._model_device())
        return all_visual.unsqueeze(0)

    @torch.no_grad()
    def encode_prompt(self, prompt: str) -> Tensor:
        """Encode prompt text tokens in the language embedding space."""

        tokenized = self.processor.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True,
        )
        tokenized = {k: v.to(self._model_device()) for k, v in tokenized.items()}
        embed_layer = self.model.get_input_embeddings()
        embeddings = embed_layer(tokenized["input_ids"])
        return embeddings

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        media_inputs: Any = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> str:
        """Standard generation path without explicit slot injection."""

        if media_inputs is not None:
            inputs = self._prepare_mm_inputs(prompt=prompt, media_inputs=media_inputs)
        else:
            tokenized = self.processor.tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=True,
            )
            inputs = {k: v.to(self._model_device()) for k, v in tokenized.items()}

        input_len = int(inputs["input_ids"].shape[1])
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0.0,
            temperature=temperature if temperature > 0.0 else None,
            top_p=top_p if temperature > 0.0 else None,
            pad_token_id=self.processor.tokenizer.pad_token_id,
        )
        return self.processor.decode(outputs[0, input_len:], skip_special_tokens=True)

    def _render_textual_slot_memory(
        self, slot_names: Optional[Sequence[str]], selected_slots: Tensor
    ) -> str:
        names = list(slot_names) if slot_names is not None else []
        lines = ["[NSSM Dynamic Memory Slots]"]
        if selected_slots.ndim == 3:
            selected_slots = selected_slots[0]
        for idx in range(selected_slots.size(0)):
            stat = float(selected_slots[idx].norm().item())
            if idx < len(names):
                lines.append(f"Slot-{idx:03d}: {names[idx]} (norm={stat:.3f})")
            else:
                lines.append(f"Slot-{idx:03d}: visual evidence cluster (norm={stat:.3f})")
        lines.append("Use these memory slots as the only visual working memory.")
        return "\n".join(lines)

    @torch.no_grad()
    def _generate_with_slot_prefix_embeddings(
        self,
        prompt: str,
        selected_slots: Tensor,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        tokenized = self.processor.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True,
        )
        tokenized = {k: v.to(self._model_device()) for k, v in tokenized.items()}

        if selected_slots.ndim == 2:
            selected_slots = selected_slots.unsqueeze(0)
        if selected_slots.ndim != 3:
            raise ValueError(
                f"selected_slots must be [K, D] or [B, K, D], got {selected_slots.shape}"
            )
        if selected_slots.size(0) != 1:
            raise ValueError("Current implementation expects batch size 1.")
        if selected_slots.size(-1) != self.hidden_size:
            raise ValueError(
                f"Slot hidden size mismatch: expected {self.hidden_size}, "
                f"got {selected_slots.size(-1)}"
            )

        prompt_embeds = self.model.get_input_embeddings()(tokenized["input_ids"])
        slot_embeds = selected_slots.to(prompt_embeds.device, prompt_embeds.dtype)
        combined_embeds = torch.cat([slot_embeds, prompt_embeds], dim=1)

        slot_len = slot_embeds.size(1)
        pad_id = self.processor.tokenizer.pad_token_id
        synthetic_prefix_ids = torch.full(
            (1, slot_len),
            fill_value=pad_id,
            device=prompt_embeds.device,
            dtype=tokenized["input_ids"].dtype,
        )
        synthetic_input_ids = torch.cat(
            [synthetic_prefix_ids, tokenized["input_ids"]], dim=1
        )
        attention_mask = torch.ones_like(synthetic_input_ids)
        input_len = int(synthetic_input_ids.size(1))

        outputs = self.model.generate(
            input_ids=synthetic_input_ids,
            inputs_embeds=combined_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0.0,
            temperature=temperature if temperature > 0.0 else None,
            top_p=top_p if temperature > 0.0 else None,
            pad_token_id=self.processor.tokenizer.pad_token_id,
        )
        return self.processor.decode(outputs[0, input_len:], skip_special_tokens=True)

    @torch.no_grad()
    def generate_with_selected_slots(
        self,
        prompt: str,
        selected_slots: Tensor,
        slot_names: Optional[Sequence[str]] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> str:
        """Generate final response using routed NSSM slots.

        Preferred strategy:
            Prefix selected slot embeddings directly into the language embedding
            stream before prompt tokens.

        Fallback strategy:
            If a backend-specific generation path rejects custom embeddings,
            inject a textual memory summary synthesized from slot names.
        """

        if selected_slots.numel() == 0:
            return self.generate(
                prompt=prompt,
                media_inputs=None,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

        if not self.force_textual_fallback:
            try:
                return self._generate_with_slot_prefix_embeddings(
                    prompt=prompt,
                    selected_slots=selected_slots,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
            except Exception as exc:  # pragma: no cover - fallback path for portability.
                LOGGER.warning(
                    "Slot-prefix embedding generation failed (%s). "
                    "Falling back to textual memory prompt.",
                    exc,
                )
        else:
            LOGGER.info("Skipping slot-prefix embeddings and forcing textual fallback.")
        memory_text = self._render_textual_slot_memory(
            slot_names=slot_names,
            selected_slots=selected_slots,
        )
        enriched_prompt = (
            f"{memory_text}\n\n"
            f"[User Question]\n{prompt}\n\n"
            "Answer based only on the routed memory slots."
        )
        return self.generate(
            prompt=enriched_prompt,
            media_inputs=None,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
