"""End-to-end NSSM inference engine.

Core research claim:
    Named Semantic Slot Memory (NSSM) performs memory compression at inference
    time conditioned on the *current* query, instead of using an offline fixed
    visual compressor.

This engine explicitly follows a five-step cognitive pipeline:
1. Perception: extract base visual tokens from raw media.
2. Dynamic aggregation: construct query-aware slots from visual tokens.
3. Explicit naming: convert slots into short textual metadata.
4. Routing and injection: select top-K slots by query relevance.
5. System-2 reasoning: answer using only the routed working-memory slots.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml

from src.eval.metrics import (
    aggregate_metrics,
    compute_accuracy,
    measure_peak_vram_gb,
    reset_peak_vram_stats,
)
from src.eval.mmlongbench_loader import load_mmlongbench_samples
from src.eval.output_sanitizer import sanitize_answer
from src.models import BackendConfig, build_backend
from src.models.dynamic_slot_attn import QueryAwareSlotAggregator
from src.pipeline.memory_router import NameAwareMemoryRouter
from src.pipeline.slot_namer import PrototypeSlotNamer

LOGGER = logging.getLogger(__name__)


@dataclass
class NSSMHyperParams:
    """NSSM-specific hyper-parameters."""

    max_visual_tokens_raw: int = 100000
    num_dynamic_slots: int = 256
    router_top_k: int = 32
    aggregator_num_heads: int = 8
    aggregator_dropout: float = 0.0
    enable_llm_slot_refine: bool = False
    slot_refine_group_size: int = 8
    routing_alpha_text_name: float = 0.6


class NSSMSystem:
    """Main orchestration class for NSSM inference."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        model_cfg = config.get("model", {})
        nssm_cfg = config.get("nssm", {})

        self.hparams = NSSMHyperParams(
            max_visual_tokens_raw=int(nssm_cfg.get("max_visual_tokens_raw", 100000)),
            num_dynamic_slots=int(nssm_cfg.get("num_dynamic_slots", 256)),
            router_top_k=int(nssm_cfg.get("router_top_k", 32)),
            aggregator_num_heads=int(nssm_cfg.get("aggregator_num_heads", 8)),
            aggregator_dropout=float(nssm_cfg.get("aggregator_dropout", 0.0)),
            enable_llm_slot_refine=bool(nssm_cfg.get("enable_llm_slot_refine", False)),
            slot_refine_group_size=int(nssm_cfg.get("slot_refine_group_size", 8)),
            routing_alpha_text_name=float(nssm_cfg.get("routing_alpha_text_name", 0.6)),
        )

        backend_config = BackendConfig(
            model_name=str(model_cfg.get("model_name", "Qwen/Qwen2.5-VL-7B-Instruct")),
            model_local_path=model_cfg.get("model_local_path"),
            precision=str(model_cfg.get("precision", "bfloat16")),
            device_map=str(model_cfg.get("device_map", "auto")),
            use_flash_attention_2=bool(model_cfg.get("use_flash_attention_2", True)),
            torch_compile=bool(model_cfg.get("torch_compile", False)),
            trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
            image_resize=(
                float(model_cfg.get("image_resize"))
                if model_cfg.get("image_resize") is not None
                else None
            ),
            max_image_num=(
                int(model_cfg.get("max_image_num"))
                if model_cfg.get("max_image_num") is not None
                else None
            ),
            force_textual_fallback=bool(model_cfg.get("force_textual_fallback", False)),
        )
        backend_name = str(model_cfg.get("backend", "qwen"))
        self.backend = build_backend(backend_name=backend_name, config=backend_config)

        self.aggregator = QueryAwareSlotAggregator(
            hidden_size=self.backend.hidden_size,
            num_slots=self.hparams.num_dynamic_slots,
            num_heads=self.hparams.aggregator_num_heads,
            dropout=self.hparams.aggregator_dropout,
        )
        self.slot_namer = PrototypeSlotNamer(max_label_words=6)
        self.router = NameAwareMemoryRouter(
            top_k=self.hparams.router_top_k,
            alpha_text_name=self.hparams.routing_alpha_text_name,
        )
        self._aggregator_ready = False
        self._compile_aggregator = bool(model_cfg.get("compile_aggregator", False))

        LOGGER.info(
            "Initialized NSSM with backend=%s, slots=%d, top_k=%d",
            backend_name,
            self.hparams.num_dynamic_slots,
            self.hparams.router_top_k,
        )

    def _prepare_aggregator(self, reference_tensor: torch.Tensor) -> None:
        if self._aggregator_ready:
            return
        self.aggregator.to(
            device=reference_tensor.device,
            dtype=reference_tensor.dtype,
        )
        if self._compile_aggregator:
            try:
                self.aggregator = torch.compile(self.aggregator)
                LOGGER.info("Enabled torch.compile for NSSM aggregator.")
            except Exception as exc:  # pragma: no cover - runtime fallback
                LOGGER.warning("torch.compile failed for NSSM aggregator: %s", exc)
        self.aggregator.eval()
        self._aggregator_ready = True

    @torch.no_grad()
    def generate_response(
        self,
        media_inputs: Any,
        prompt: str,
        return_debug: bool = False,
    ) -> Dict[str, Any]:
        """Run all five NSSM stages for one query.

        Args:
            media_inputs: Image/video payload accepted by backend.
            prompt: User query.
            return_debug: If true, include intermediate diagnostics.
        """

        stage_latency_ms: Dict[str, float] = {}
        reset_peak_vram_stats()
        start_total = time.perf_counter()

        # Step 1: Perception.
        t0 = time.perf_counter()
        base_visual_tokens = self.backend.extract_visual_tokens(media_inputs, prompt)
        if base_visual_tokens.size(1) > self.hparams.max_visual_tokens_raw:
            base_visual_tokens = base_visual_tokens[:, : self.hparams.max_visual_tokens_raw, :]
        stage_latency_ms["step1_perception"] = (time.perf_counter() - t0) * 1000.0

        generation_cfg = self.config.get("generation", {})
        if base_visual_tokens.size(1) == 0:
            t0 = time.perf_counter()
            raw_answer = self.backend.generate(
                prompt=prompt,
                media_inputs=None,
                max_new_tokens=int(generation_cfg.get("max_new_tokens", 256)),
                temperature=float(generation_cfg.get("temperature", 0.0)),
                top_p=float(generation_cfg.get("top_p", 1.0)),
            )
            stage_latency_ms["step2_dynamic_aggregation"] = 0.0
            stage_latency_ms["step3_explicit_naming"] = 0.0
            stage_latency_ms["step4_routing"] = 0.0
            stage_latency_ms["step5_system2_reasoning"] = (time.perf_counter() - t0) * 1000.0
            sanitized = sanitize_answer(prompt=prompt, answer=raw_answer)
            total_latency_ms = (time.perf_counter() - start_total) * 1000.0
            vram_peak_gb = measure_peak_vram_gb()
            output = {
                "answer": sanitized.answer,
                "answer_raw": raw_answer,
                "sanitization_mode": sanitized.mode,
                "latency_ms": float(total_latency_ms),
                "vram_peak_gb": float(vram_peak_gb),
                "selected_slot_ids": [],
                "selected_slot_names": [],
                "router_scores": [],
            }
            if return_debug:
                output["debug"] = {
                    "stage_latency_ms": {k: float(v) for k, v in stage_latency_ms.items()},
                    "num_raw_visual_tokens": 0,
                    "num_dynamic_slots": 0,
                    "num_selected_slots": 0,
                    "slot_name_confidence": [],
                    "bypass_reason": "no_visual_tokens",
                }
            return output

        # Step 2: Dynamic query-aware slot construction.
        t0 = time.perf_counter()
        prompt_embeds = self.backend.encode_prompt(prompt)
        self._prepare_aggregator(prompt_embeds)
        dynamic_slots = self.aggregator(
            visual_tokens=base_visual_tokens.to(prompt_embeds.device, prompt_embeds.dtype),
            text_prompt_embeds=prompt_embeds,
        )
        stage_latency_ms["step2_dynamic_aggregation"] = (time.perf_counter() - t0) * 1000.0

        # Step 3: Explicit slot naming.
        t0 = time.perf_counter()
        slot_metadata = self.slot_namer.name_slots(
            dynamic_slots=dynamic_slots,
            prompt_embeds=prompt_embeds,
            prompt_text=prompt,
            backend=self.backend,
            enable_llm_refine=self.hparams.enable_llm_slot_refine,
            llm_group_size=self.hparams.slot_refine_group_size,
        )
        stage_latency_ms["step3_explicit_naming"] = (time.perf_counter() - t0) * 1000.0

        # Step 4: Name-aware routing.
        t0 = time.perf_counter()
        router_output = self.router.route(
            prompt_text=prompt,
            prompt_embeds=prompt_embeds,
            dynamic_slots=dynamic_slots,
            slot_metadata=slot_metadata,
            backend=self.backend,
        )
        stage_latency_ms["step4_routing"] = (time.perf_counter() - t0) * 1000.0

        # Step 5: System-2 reasoning with routed slots only.
        t0 = time.perf_counter()
        raw_answer = self.backend.generate_with_selected_slots(
            prompt=prompt,
            selected_slots=router_output.selected_slots,
            slot_names=[item.name for item in router_output.selected_metadata],
            max_new_tokens=int(generation_cfg.get("max_new_tokens", 256)),
            temperature=float(generation_cfg.get("temperature", 0.0)),
            top_p=float(generation_cfg.get("top_p", 1.0)),
        )
        stage_latency_ms["step5_system2_reasoning"] = (time.perf_counter() - t0) * 1000.0
        sanitized = sanitize_answer(prompt=prompt, answer=raw_answer)

        end_total = time.perf_counter()
        total_latency_ms = (end_total - start_total) * 1000.0
        vram_peak_gb = measure_peak_vram_gb()

        output: Dict[str, Any] = {
            "answer": sanitized.answer,
            "answer_raw": raw_answer,
            "sanitization_mode": sanitized.mode,
            "latency_ms": float(total_latency_ms),
            "vram_peak_gb": float(vram_peak_gb),
            "selected_slot_ids": router_output.indices,
            "selected_slot_names": [item.name for item in router_output.selected_metadata],
            "router_scores": [float(x) for x in router_output.scores],
        }
        if return_debug:
            output["debug"] = {
                "stage_latency_ms": {k: float(v) for k, v in stage_latency_ms.items()},
                "num_raw_visual_tokens": int(base_visual_tokens.size(1)),
                "num_dynamic_slots": int(dynamic_slots.size(1)),
                "num_selected_slots": int(router_output.selected_slots.size(1)),
                "slot_name_confidence": [
                    float(item.confidence) for item in router_output.selected_metadata
                ],
            }
        return output

    def evaluate_mmlongbench(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate NSSM on a selected MMLongBench file."""

        data_cfg = config.get("data", {})
        data_root = str(data_cfg.get("data_root"))
        image_root = str(data_cfg.get("image_root"))
        dataset_file = str(data_cfg.get("dataset_file"))
        max_samples = data_cfg.get("max_samples")
        max_samples = int(max_samples) if max_samples is not None else None

        samples = load_mmlongbench_samples(
            data_root=data_root,
            image_root=image_root,
            dataset_file=dataset_file,
            max_samples=max_samples,
        )
        LOGGER.info("Loaded %d samples from %s", len(samples), dataset_file)

        records: List[Dict[str, Any]] = []
        for idx, sample in enumerate(samples):
            media_inputs = {"images": sample.image_paths}
            out = self.generate_response(
                media_inputs=media_inputs,
                prompt=sample.prompt,
                return_debug=bool(config.get("runtime", {}).get("save_debug", True)),
            )
            out["sample_id"] = sample.sample_id
            out["gold_answer"] = sample.answer
            out["accuracy"] = compute_accuracy(out["answer"], sample.answer)
            records.append(out)
            LOGGER.info(
                "[%d/%d] sample=%s acc=%.3f latency=%.1fms vram=%.2fGB",
                idx + 1,
                len(samples),
                sample.sample_id,
                out["accuracy"],
                out["latency_ms"],
                out["vram_peak_gb"],
            )

        return {
            "config": config,
            "metrics": aggregate_metrics(records),
            "records": records,
        }


def _load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _dump_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NSSM MMLongBench inference runner")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/exp_mmlongbench_128k.yaml",
        help="Path to YAML experiment config.",
    )
    parser.add_argument(
        "--dataset_file",
        type=str,
        default=None,
        help="Optional override for data.dataset_file.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional override for data.max_samples.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Optional override for runtime.output_file.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    args = parse_args()
    config = _load_config(args.config)

    if args.dataset_file is not None:
        config.setdefault("data", {})["dataset_file"] = args.dataset_file
    if args.max_samples is not None:
        config.setdefault("data", {})["max_samples"] = args.max_samples
    if args.output_file is not None:
        config.setdefault("runtime", {})["output_file"] = args.output_file

    seed = int(config.get("hardware", {}).get("seed", 42))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    system = NSSMSystem(config=config)
    result = system.evaluate_mmlongbench(config=config)
    output_file = str(
        config.get("runtime", {}).get(
            "output_file",
            str(Path("outputs") / "nssm_mmlongbench_eval.json"),
        )
    )
    _dump_json(output_file, result)
    LOGGER.info("Evaluation completed. Results saved to %s", output_file)
    LOGGER.info("Aggregate metrics: %s", result["metrics"])


if __name__ == "__main__":
    main()
