"""Shared manifest helpers for official full MMLongBench runs."""

from __future__ import annotations

import math
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import yaml


TASK_CONFIG_MAP = OrderedDict(
    [
        ("vrag", "vrag_all.yaml"),
        ("vh", "vh_all.yaml"),
        ("mm_niah_text", "mm_niah_text_all.yaml"),
        ("mm_niah_image", "mm_niah_image_all.yaml"),
        ("icl", "icl_all.yaml"),
        ("summ", "summ_all.yaml"),
        ("docqa", "docqa_all.yaml"),
    ]
)

DEFAULT_FULL_TASKS = list(TASK_CONFIG_MAP.keys())
DEFAULT_LENGTHS = [8, 16, 32, 64, 128]

DATASET_METRIC_PRIORITY = {
    "infoseek": ("sub_em",),
    "viquae": ("sub_em",),
    "vh_single": ("acc",),
    "vh_multi": ("acc",),
    "mm_niah_retrieval-text": ("sub_em",),
    "mm_niah_counting-text": ("soft_acc",),
    "mm_niah_reasoning-text": ("sub_em",),
    "mm_niah_retrieval-image": ("mc_acc",),
    "mm_niah_counting-image": ("soft_acc",),
    "mm_niah_reasoning-image": ("mc_acc",),
    "cars196": ("cls_acc",),
    "food101": ("cls_acc",),
    "inat2021": ("cls_acc",),
    "sun397": ("cls_acc",),
    "gov-report": ("gpt4-flu-f1", "gpt4-f1", "rougeLsum", "rougeL"),
    "multi-lexsum": ("gpt4-flu-f1", "gpt4-f1", "rougeLsum", "rougeL"),
    "longdocurl": ("doc_qa_llm", "doc_qa"),
    "mmlongdoc": ("doc_qa_llm", "doc_qa"),
    "slidevqa": ("doc_qa_llm", "doc_qa"),
}

OFFICIAL_AGGREGATES = OrderedDict(
    [
        ("VRAG", ["infoseek", "viquae"]),
        ("mm_niah_retrieval", ["mm_niah_retrieval-text", "mm_niah_retrieval-image"]),
        ("mm_niah_counting", ["mm_niah_counting-text", "mm_niah_counting-image"]),
        ("mm_niah_reasoning", ["mm_niah_reasoning-text", "mm_niah_reasoning-image"]),
        (
            "NIAH",
            [
                "vh_single",
                "vh_multi",
                "mm_niah_retrieval",
                "mm_niah_counting",
                "mm_niah_reasoning",
            ],
        ),
        ("ICL", ["cars196", "food101", "inat2021", "sun397"]),
        ("Summ", ["gov-report", "multi-lexsum"]),
        ("DocVQA", ["longdocurl", "mmlongdoc", "slidevqa"]),
        ("Avg", ["VRAG", "NIAH", "ICL", "Summ", "DocVQA"]),
    ]
)


def _parse_csv(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _parse_bool(raw: Any, default: bool = False) -> bool:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    normalized = str(raw).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    return default


def _extract_length_k(text: str) -> Optional[int]:
    match = re.search(r"_K(\d+)", text)
    if match:
        return int(match.group(1))
    return None


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@dataclass(frozen=True)
class RuntimeSettings:
    generation_min_length: int = 0
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    seed: int = 42
    max_test_samples_override: Optional[int] = None


@dataclass(frozen=True)
class DatasetRunSpec:
    task: str
    config_name: str
    dataset: str
    test_file: str
    length_k: int
    input_max_length: int
    generation_max_length: int
    use_chat_template: bool
    max_test_samples: Optional[int]

    @property
    def test_name(self) -> str:
        return Path(self.test_file).stem

    @property
    def is_summ(self) -> bool:
        return self.task == "summ"

    @property
    def is_text_only(self) -> bool:
        return self.task == "mm_niah_text"

    @property
    def job_id(self) -> str:
        safe_dataset = re.sub(r"[^A-Za-z0-9._-]+", "_", self.dataset)
        return f"{self.task}__{safe_dataset}__K{self.length_k}"

    def effective_max_test_samples(self, runtime: RuntimeSettings) -> Optional[int]:
        if runtime.max_test_samples_override is not None:
            return int(runtime.max_test_samples_override)
        return self.max_test_samples

    def expected_output_name(self, runtime: RuntimeSettings) -> str:
        max_samples = self.effective_max_test_samples(runtime)
        return (
            f"{self.dataset}_{self.test_name}_in{self.input_max_length}"
            f"_size{max_samples}_samp{runtime.do_sample}"
            f"max{self.generation_max_length}min{runtime.generation_min_length}"
            f"t{runtime.temperature}p{runtime.top_p}"
            f"_chat{self.use_chat_template}_{runtime.seed}.json"
        )

    def output_prefix(self) -> str:
        return f"{self.dataset}_{self.test_name}_in{self.input_max_length}_"

    def expected_metric_names(self) -> Sequence[str]:
        return DATASET_METRIC_PRIORITY.get(self.dataset, ())


def get_default_bench_root() -> Path:
    return Path(__file__).resolve().parents[3] / "MMLongBench"


def iter_specs(
    bench_root: Optional[Path] = None,
    task_list: Optional[Sequence[str]] = None,
    length_list: Optional[Sequence[int]] = None,
) -> List[DatasetRunSpec]:
    bench_root = Path(bench_root) if bench_root is not None else get_default_bench_root()
    task_names = list(task_list or DEFAULT_FULL_TASKS)
    wanted_lengths = set(int(item) for item in (length_list or DEFAULT_LENGTHS))

    specs: List[DatasetRunSpec] = []
    for task in task_names:
        config_name = TASK_CONFIG_MAP[task]
        config = _read_yaml(bench_root / "configs" / config_name)
        datasets = _parse_csv(config.get("datasets"))
        test_files = _parse_csv(config.get("test_files"))
        input_lengths = [int(item) for item in _parse_csv(config.get("input_max_length"))]
        gen_lengths = [int(item) for item in _parse_csv(config.get("generation_max_length"))]
        use_chat_template = _parse_bool(config.get("use_chat_template"), default=True)
        max_test_samples = config.get("max_test_samples")
        max_test_samples = int(max_test_samples) if max_test_samples is not None else None

        for dataset, test_file, input_len, gen_len in zip(
            datasets,
            test_files,
            input_lengths,
            gen_lengths,
        ):
            length_k = _extract_length_k(test_file)
            if length_k is None:
                length_k = int(math.floor(int(input_len) / 1024))
            if length_k not in wanted_lengths:
                continue
            specs.append(
                DatasetRunSpec(
                    task=task,
                    config_name=config_name,
                    dataset=dataset,
                    test_file=test_file,
                    length_k=length_k,
                    input_max_length=int(input_len),
                    generation_max_length=int(gen_len),
                    use_chat_template=use_chat_template,
                    max_test_samples=max_test_samples,
                )
            )
    specs.sort(key=lambda item: (-item.length_k, item.task, item.dataset))
    return specs


def specs_by_job_id(specs: Iterable[DatasetRunSpec]) -> Dict[str, DatasetRunSpec]:
    return {spec.job_id: spec for spec in specs}
