# NSSM Research 使用说明（中文）

本项目是一个论文导向的研究代码骨架，目标是实现 **NSSM (Named Semantic Slot Memory)**：  
在推理时（而不是离线预训练）根据当前用户 Prompt 动态构建视觉语义槽，并进行显式命名与路由，再用于最终回答。

当前默认后端为 **Qwen2.5-VL-7B-Instruct**，并已设计统一 backend 协议，便于后续无缝切换到其他模型。

## 1. 目录结构

```text
nssm_research/
├── configs/
│   ├── exp_mmlongbench_128k.yaml
│   └── exp_mmlongbench_vision_4gpu.yaml
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── dynamic_slot_attn.py
│   │   └── qwen_nssm_wrapper.py
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── slot_namer.py
│   │   ├── memory_router.py
│   │   └── inference_engine.py
│   └── eval/
│       ├── mmlongbench_loader.py
│       ├── mmlongbench_report.py
│       └── metrics.py
├── scripts/
│   ├── run_mmlongbench.sh
│   └── run_mmlongbench_vision_4gpu.sh
└── requirements.txt
```

## 2. 数据与模型路径约定

- 参考文献（供方法对齐）：
  - `/workspace/zhuo/long_visual_context/doc/2505.10610v3.pdf`
  - `/workspace/zhuo/long_visual_context/doc/2603.01143v1.pdf`
- MMLongBench 数据：
  - 文本标注：`/workspace/zhuo/long_visual_context/data/mmlb_data`
  - 图片数据：`/workspace/zhuo/long_visual_context/data/mmlb_image`
- 本地 Qwen 模型（默认优先）：
  - `/workspace/helandi/model/Qwen-7b/Qwen2.5-VL-7B-Instruct/`

## 3. 环境安装

进入项目目录并安装依赖：

```bash
cd /workspace/zhuo/long_visual_context/nssm_research
python -m pip install -r requirements.txt
```

如果你使用 conda 环境，建议先激活对应环境再安装。

## 4. 快速开始（先跑通 Qwen）

### 4.1 默认运行（accelerate 双卡）

```bash
cd /workspace/zhuo/long_visual_context/nssm_research
bash scripts/run_mmlongbench.sh
```

该命令默认读取：

- 配置文件：`configs/exp_mmlongbench_128k.yaml`
- 启动入口：`src/pipeline/inference_engine.py`

### 4.2 覆盖数据集与样本数（调试常用）

```bash
cd /workspace/zhuo/long_visual_context/nssm_research
bash scripts/run_mmlongbench.sh \
  configs/exp_mmlongbench_128k.yaml \
  --dataset_file documentQA/mmlongdoc_K4.jsonl \
  --max_samples 2
```

### 4.3 输出结果

默认输出由配置项 `runtime.output_file` 指定，当前默认路径为：

`/workspace/zhuo/long_visual_context/nssm_research/outputs/mmlongbench_nssm_predictions.jsonl`

输出中包含：

- `answer`：模型回答
- `accuracy`：样本准确率（EM 风格）
- `latency_ms`：总耗时
- `vram_peak_gb`：峰值显存
- `selected_slot_ids / selected_slot_names / router_scores`：NSSM 路由可解释信息
- `debug.stage_latency_ms`：五阶段细粒度耗时

### 4.4 四卡视觉版 MMLongBench 完整验证

这一条链路只在 `nssm_research/` 下新增脚本与汇总器，不改 `MMLongBench/` 源码；实际评测仍复用官方 `MMLongBench/scripts/eval_task_manager.py` 和 `--use_nssm` 入口。

默认覆盖的视觉任务为：

- `vrag`
- `vh`
- `mm_niah_image`
- `icl`
- `docqa`

默认测试长度为：

- `8`
- `16`
- `32`
- `64`
- `128`

推荐直接运行：

```bash
cd /workspace/zhuo/long_visual_context/nssm_research
bash scripts/run_mmlongbench_vision_4gpu.sh
```

该脚本的默认行为：

- 使用 `conda run -n mmlongbench`
- 使用四张卡 `0,1,2,3`
- 使用配置 `configs/exp_mmlongbench_vision_4gpu.yaml`
- 输出目录为 `nssm_research/outputs/mmlongbench_vision_4gpu`
- `docqa_llm_judge=False`

常用覆盖方式：

```bash
CONDA_ENV=mmlongbench \
GPU_LIST=0,1,2,3 \
MODEL_NAME=/workspace/helandi/model/Qwen-7b/Qwen2.5-VL-7B-Instruct/ \
RESULT_BASE_PATH=/workspace/zhuo/long_visual_context/nssm_research/outputs/mmlongbench_vision_4gpu \
bash scripts/run_mmlongbench_vision_4gpu.sh --overwrite
```

注意：

- 脚本会先切换到同级 `MMLongBench/` 目录，再调用官方 `eval_task_manager.py`，这是为了兼容其内部对 `eval.py` 的相对路径调用。
- `docqa` 默认关闭 LLM judge，因此不需要额外 API key；对应结果按本地官方非 LLM 指标汇总。
- `eval_task_manager.py` 会在结果目录下自动创建以模型名为后缀的子目录，例如 `Qwen2.5-VL-7B-Instruct/`。

### 4.5 结果汇总与实验总结

四卡实验结束后，使用下面的命令生成 `summary.json` 和 `summary.md`：

```bash
cd /workspace/zhuo/long_visual_context/nssm_research
conda run -n mmlongbench python -m src.eval.mmlongbench_report \
  --result_dir /workspace/zhuo/long_visual_context/nssm_research/outputs/mmlongbench_vision_4gpu/Qwen2.5-VL-7B-Instruct \
  --output_dir /workspace/zhuo/long_visual_context/nssm_research/outputs/mmlongbench_vision_4gpu/report
```

汇总结果会包含：

- 总体完成度检查：哪些任务或长度缺失
- 分任务、分长度的统一 `score_pct`
- 代表性错误样例
- 面向 NSSM 的改进建议

输出文件：

- `summary.json`：机器可读
- `summary.md`：人工阅读版实验总结

## 5. 核心配置说明

编辑 `configs/exp_mmlongbench_128k.yaml`：

### 5.1 模型配置（model）

- `backend`: 当前为 `qwen`
- `model_name`: HF 名称回退
- `model_local_path`: 本地路径优先
- `precision`: 推荐 `bfloat16`
- `device_map`: 推荐 `auto`
- `use_flash_attention_2`: 推荐 `true`

### 5.2 NSSM 配置（nssm）

- `max_visual_tokens_raw`: 原始视觉 token 上限（默认 100000）
- `num_dynamic_slots`: 动态槽数量（默认 256）
- `router_top_k`: 路由后注入槽数（默认 32）
- `enable_llm_slot_refine`: 是否用 LLM 重写槽名称（默认 false）
- `routing_alpha_text_name`: 路由时名称相似度权重

### 5.3 数据配置（data）

- `data_root`: `mmlb_data` 根目录
- `image_root`: `mmlb_image` 根目录
- `dataset_file`: 如 `documentQA/mmlongdoc_K128.jsonl`
- `max_samples`: 调试时建议先设小值

## 6. NSSM 推理流程（对应代码）

主入口：`src/pipeline/inference_engine.py` 中的 `NSSMSystem.generate_response(...)`

1. Step 1 感知：提取 `base_visual_tokens`
2. Step 2 动态聚合：`QueryAwareSlotAggregator` 构建 `dynamic_slots`
3. Step 3 显式命名：`PrototypeSlotNamer` 生成 slot labels
4. Step 4 路由：`NameAwareMemoryRouter` 选 Top-K
5. Step 5 System-2：仅用 Top-K slots 生成最终回答

## 7. 如何切换到其他模型（无缝扩展）

你不需要改 pipeline 主流程，只需要实现一个新的 backend 适配器：

1. 在 `src/models/` 新建例如 `internvl_nssm_wrapper.py`
2. 实现 `BaseVLMBackend` 的 5 个核心接口：
   - `hidden_size`
   - `extract_visual_tokens`
   - `encode_prompt`
   - `generate`
   - `generate_with_selected_slots`
3. 在 `src/models/__init__.py` 的 `build_backend(...)` 里注册新 backend
4. 把配置 `model.backend` 改为你的新 backend 名称

这样 `NSSMSystem` 无需改动即可复用整条链路。

## 8. 常见问题

### Q1: 启动时报 `ModuleNotFoundError: torch`

说明当前 Python 环境未安装 PyTorch。先执行：

```bash
python -m pip install -r requirements.txt
```

### Q2: 本地模型路径不存在

会自动回退到 `model_name`（HF Hub 名称）。如果你是离线环境，请确保 `model_local_path` 正确。

### Q3: 显存不足

优先调整：

- `data.max_samples`（先小样本调试）
- `nssm.num_dynamic_slots`（例如 256 -> 128）
- `nssm.router_top_k`（例如 32 -> 16）

### Q4: 四卡脚本跑完后，汇总器提示有缺失任务

先检查对应目录：

- `outputs/mmlongbench_vision_4gpu/<model_tag>/<task>_<length>/error.log`
- `outputs/mmlongbench_vision_4gpu/<model_tag>/<task>_<length>/stdout.log`

再重新跑：

```bash
cd /workspace/zhuo/long_visual_context/nssm_research
bash scripts/run_mmlongbench_vision_4gpu.sh --overwrite
```

---

如果你下一步要做“可训练版 NSSM（如 LoRA 微调聚合器）”，建议先在当前推理骨架上新增训练脚本，而不是改动现有推理主干，保证论文复现实验稳定性。
