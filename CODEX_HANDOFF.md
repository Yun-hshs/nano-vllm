# CODEX_HANDOFF

## 1. 项目目标

本项目基于 `Yun-hshs/nano-vllm` / nano-vLLM，目标是：

- 将默认实验模型切到 `Qwen/Qwen3-4B`。
- 为 nano-vLLM 增加 speculative decoding 实验路径。
- 参考 DeepSeek DeepSpec / DSpark 的投机解码思路，先实现可验证的基础设施，再考虑 DSpark checkpoint 对接。
- 对比 baseline 与 speculative decoding 的吞吐、acceptance、target forwards、draft forwards 等指标。
- 最终在 4090 云服务器上运行并产出稳定 benchmark 结果。

当前云服务器连接：

```bash
ssh -p 23658 root@connect.cqa1.seetacloud.com
```

云端关键路径：

```bash
/root/nano-vllm
/root/autodl-tmp/huggingface/Qwen3-4B
/root/autodl-tmp/huggingface/Qwen3-0.6B
/root/autodl-tmp/nano-vllm-outputs
```

## 2. 当前开发进度

原始路线：

- 阶段 0：准备 baseline，不改核心逻辑
- 阶段 1：添加 benchmark 和结果统计
- 阶段 2：添加 speculative decoding 配置开关
- 阶段 3：实现最小 draft 生成器，不进入主推理
- 阶段 4：实现 target-only speculative verification 实验路径
- 阶段 5：接入 `engine.step()`，默认关闭，不影响原 `generate`
- 阶段 6：统计 acceptance length / target forwards / tokens/s
- 阶段 7：最后研究 DeepSpec / DSpark checkpoint 对接

当前实际进度：

- 阶段 0-6 已完成。
- 已完成 6.5：HF draft model speculative path，但性能差，已保留为实验路径。
- 已完成 6.6：HF draft KV cache，但性能仍差。
- 已完成 6.7：nano-vLLM native draft runner。
- 已完成 6.8：draft decode CUDA graph。
- 已完成 6.9：graph replay 统计和 k sweep 汇总。
- 已完成 6.10：target verification greedy fast path + 分段耗时统计。
- 已完成 6.10.1：profiling-only 细分耗时统计，并确认 k9 当前最佳。
- 6.11 target verification graph 尝试后已回退。
- 6.11.2 draft cache repair 尝试后已回退。
- 6.10.2 draft runner CUDA graph fusion 尝试后已回退。

当前应视为 **6.10.1 稳定版本**：native nano draft + draft decode CUDA graph + greedy target verification + profiling-only 细分耗时统计。当前最佳 speculative length 是 **k9**。

## 3. 已完成模块

### Benchmark 和统计

- `bench.py`
  - 支持 Qwen3-4B 默认 benchmark。
  - 支持 `--enable-engine-speculative`。
  - 支持 `--draft-type ngram|hf|nano`。
  - 支持 `--draft-model`。
  - 支持 `--gpu-memory-utilization`。
  - 支持 `--temperature`。
  - 支持 `--speculative-greedy-temperature`。
  - 输出 JSON，包括：
    - `proposed_tokens`
    - `accepted_tokens`
    - `draft_forwards`
    - `draft_rebuilds`
    - `draft_cached_steps`
    - `draft_graph_enabled`
    - `draft_graph_replays`
    - `target_forwards`
    - `target_greedy_forwards`
    - `target_greedy_tokens`
    - `draft_propose_ms`
    - `draft_prefill_ms`
    - `draft_cache_extend_ms`
    - `draft_decode_graph_ms`
    - `draft_decode_eager_ms`
    - `draft_token_select_ms`
    - `target_verify_ms`
    - `target_forward_ms`
    - `target_argmax_ms`
    - `target_compare_ms`
    - `append_tokens_ms`
    - `speculative_step_ms`
    - `acceptance_lengths`

### Speculative 配置和工具

- `nanovllm/speculative.py`
  - `DSparkSpeculativeConfig`
  - `SpeculativeRuntimeConfig`
  - `DraftProposal`
  - `VerificationResult`
  - `SpeculativeStats`
  - `BenchmarkResult`
  - `compare_benchmarks`
  - `format_benchmark_comparison`
  - `benchmark_result_from_dict`
  - `SpeculativeDecoder`
  - `NGramDraftGenerator`
  - `HFDraftGenerator`
  - target-only speculative experiment helpers
  - `should_use_greedy_verification`

### Engine integration

- `nanovllm/engine/llm_engine.py`
  - `step()` 根据 `speculative_config.enabled` 选择 regular/speculative path。
  - `_step_speculative()` 支持单并发、单 tensor parallel 的 experimental path。
  - 支持 `ngram`、`hf`、`nano` draft types。
  - 默认关闭 speculative，不影响原 `generate`。
  - 统计 speculative step 各阶段耗时。

### Target verification

- `nanovllm/engine/model_runner.py`
  - 新增 `run_speculative(seq, draft_token_ids)`。
  - 一次 target forward 验证 `[seq.last_token] + draft_token_ids`。
  - 低温时走 greedy argmax fast path，跳过 sampler 的 softmax/exponential sampling。
  - `temperature <= speculative_greedy_temperature` 时启用。

### Native draft runner

- `nanovllm/engine/draft_runner.py`
  - `NanoDraftGenerator`
  - 使用 nano-vLLM 的 `Qwen3ForCausalLM` 加载 draft model。
  - 单独维护 draft KV cache。
  - 支持 prefix cache 和 partial cache extension。
  - draft decode 使用 CUDA graph。
  - 禁用 draft model 上的 torch.compile wrapper，避免 Dynamo rank mismatch / recompile churn。

### 脚本

- `scripts/compare_benchmarks.py`
  - 比较 baseline 和 speculative JSON。

- `scripts/summarize_benchmarks.py`
  - 汇总 k=3/5/7 sweep。

- `scripts/target_only_speculative.py`
  - target-only speculative verification 小实验。

## 4. 正在处理的问题

当前主线目标是稳定在 6.10.1，并继续寻找安全的优化方向。

当前已知最好方向：

- 保留 k=9 作为当前最佳 speculative length。
- 当前性能大约：
  - baseline: 约 `100 tok/s`
  - k7 speculative 6.10.1: 约 `116 tok/s`
  - k9 speculative 6.10.1: 约 `127-130 tok/s`
  - k9 repeat 5 次：`127.66, 130.20, 124.47, 127.68, 128.68 tok/s`
  - k9 mean/std：`127.74 +/- 1.88 tok/s`
  - speedup: 约 `1.27x-1.30x`

不要继续推进以下三个已失败方向，除非重新设计：

- target verification prefill CUDA graph
- native draft cache repair / lazy last decode
- draft runner CUDA graph fusion（把 `compute_logits + argmax` 放进 draft decode graph）

## 5. 关键文件说明

```text
bench.py
```

Benchmark 入口。新增参数和 JSON 输出都在这里接线。

```text
nanovllm/speculative.py
```

Speculative decoding 配置、统计、benchmark result、draft generator、target-only experiment 的核心文件。

```text
nanovllm/engine/llm_engine.py
```

Engine step 接入 speculative path 的位置。重点看：

- `step`
- `_step_regular`
- `_step_speculative`
- `_append_speculative_tokens`

```text
nanovllm/engine/model_runner.py
```

Target model runner。重点看：

- `run`
- `run_model`
- `run_speculative`
- `capture_cudagraph`

```text
nanovllm/engine/draft_runner.py
```

Native nano draft model。重点看：

- `NanoDraftGenerator`
- `_capture_decode_graph`
- `propose`
- `_ensure_cache`
- `_prefill`
- `_decode_one_graph`
- `_decode_one_eager`

```text
scripts/summarize_benchmarks.py
```

k sweep markdown 汇总。

```text
tests/
```

当前有 25 个单测，主要是 source-level hook tests 和 speculative utility tests。

## 6. 当前测试状态

本地当前通过：

```bash
python3 -m unittest discover -s tests
```

最近输出：

```text
Ran 25 tests in 0.047s
OK
```

本地编译检查通过：

```bash
python3 -m compileall nanovllm bench.py scripts tests
```

最近输出为 exit 0。

注意：本地 Mac 没有 torch，因此测试主要避免 import full runtime。运行 GPU benchmark 必须在云端 4090 服务器。

## 7. 已知 bug 和错误日志

### 已修复/规避：target verification CUDA graph 非法访存

曾尝试实现 target verification prefill CUDA graph，云端报错：

```text
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with TORCH_USE_CUDA_DSA to enable device-side assertions.
```

当时发生位置：

```text
llm.generate(["Benchmark: "], SamplingParams())
...
output, num_tokens = self.step()
```

结论：

- `flash_attn_varlen_func + KV cache + block_table` 这条 prefill graph capture 路径在当前环境不安全。
- 已完全移除 target verification graph 相关代码、CLI 和统计字段。
- 不要恢复这条路径，除非做单独最小复现和隔离验证。

### 已回退：native draft cache repair / lazy decode

曾尝试 6.11.2：

- verification 后回退 draft cache 到 accepted prefix
- propose 不 decode 最后一个 draft token

云端结果明显变差：

```text
Mode: engine-speculative, Total: 255tok, Time: 2.75s, Throughput: 92.79tok/s
```

结论：

- 该优化破坏了当前 cache/next_logits 节奏，已回退。
- 不要继续沿这个 patch 增量修。

### 已回退：draft runner CUDA graph fusion

曾尝试 6.10.2：

- 将 draft decode CUDA graph 的边界从 `model(input_ids, positions)` 扩展到 `model -> compute_logits -> argmax`。
- graph 静态输出由 hidden states 改为 `next_token_ids`。
- `propose()` 内部用 GPU buffer 串联 draft token，最后一次性 `.cpu().tolist()`，避免每 token `.item()`。

云端结果变差：

```text
k7: 110.42 tok/s, draft_propose_ms 1324.76, draft_decode_graph_ms 972.89, draft_token_select_ms 0.00
k9: 111.12 tok/s, draft_propose_ms 1292.89, draft_decode_graph_ms 996.49, draft_token_select_ms 0.00
k11: 104.60 tok/s, draft_propose_ms 1682.73, draft_decode_graph_ms 1252.62, draft_token_select_ms 0.00
```

结论：

- `.item()` 同步确实被消除，但 `compute_logits + argmax` 进入 graph 后让 graph replay 明显变重，吞掉收益。
- 已回退到 6.10.1 profiling-only。
- 不要继续沿该 patch 增量修；如要重做，必须先独立微基准验证 `compute_logits` / logits projection 的成本。

### HF draft backend 性能差

HF draft without cache：

```text
k7 HF: 43.11 tok/s, acceptance 81.51%, mean accepted 5.68, target_forwards 38, speedup 0.43x
```

HF draft with cache：

```text
k7 HF cache: 37.52 tok/s, acceptance 81.51%, mean accepted 5.68, draft_forwards 303, draft_rebuilds 8, cached_steps 12791, target_forwards 38, speedup 0.37x
```

结论：

- Transformers/HF draft backend 由于 token-level overhead 太高，不适合作为当前优化主线。

## 8. 下一步任务清单

推荐下一步从安全方向开始：

1. 确认云端已经同步 6.10.1 restore 版本，并以 k9 作为当前默认最佳配置：

```bash
python bench.py \
  --model /root/autodl-tmp/huggingface/Qwen3-4B \
  --enable-engine-speculative \
  --draft-type nano \
  --draft-model /root/autodl-tmp/huggingface/Qwen3-0.6B \
  --gpu-memory-utilization 0.75 \
  --num-speculative-tokens 9 \
  --num-seqs 1 \
  --max-input-len 512 \
  --max-output-len 256 \
  --temperature 0.01 \
  --speculative-greedy-temperature 0.011 \
  --out /root/autodl-tmp/nano-vllm-outputs/bench-nano-draft-k9-6.10.1-restore.json
```

2. 如需确认稳定性，跑 k9 repeat：

```bash
for i in 1 2 3 4 5; do
  python bench.py \
    --model /root/autodl-tmp/huggingface/Qwen3-4B \
    --enable-engine-speculative \
    --draft-type nano \
    --draft-model /root/autodl-tmp/huggingface/Qwen3-0.6B \
    --gpu-memory-utilization 0.75 \
    --num-speculative-tokens 9 \
    --num-seqs 1 \
    --max-input-len 512 \
    --max-output-len 256 \
    --temperature 0.01 \
    --speculative-greedy-temperature 0.011 \
    --out /root/autodl-tmp/nano-vllm-outputs/bench-nano-draft-k9-6.10.1-repeat-${i}.json
done
```

3. repeat 汇总：

```bash
python - <<'PY'
import json
from pathlib import Path

rows = []
for i in range(1, 6):
    path = Path(f"/root/autodl-tmp/nano-vllm-outputs/bench-nano-draft-k9-6.10.1-repeat-{i}.json")
    data = json.loads(path.read_text())
    tps = data["output_tokens"] / data["elapsed_seconds"]
    rows.append(tps)
    print(f"run {i}: {tps:.2f} tok/s")

mean = sum(rows) / len(rows)
var = sum((value - mean) ** 2 for value in rows) / len(rows)
print(f"\nmean: {mean:.2f} tok/s")
print(f"min:  {min(rows):.2f} tok/s")
print(f"max:  {max(rows):.2f} tok/s")
print(f"std:  {var ** 0.5:.2f} tok/s")
PY
```

4. 下一轮优化建议：

- 不碰 target prefill graph。
- 不碰 partial reject draft cache repair。
- 不碰 draft runner CUDA graph fusion。
- k9 作为默认最佳点，后续优化要以 repeat 均值为准，不要只看单次峰值。
- 然后判断是否值得做：
  - 更大的输出 token 数 benchmark，避免 255 token 小样本波动。
  - 多 prompt / 低并发场景 sweep。
  - 独立微基准拆解 draft `compute_logits` / logits projection 成本，不直接接 engine。

5. DSpark / DeepSpec checkpoint 对接仍是后续阶段 7，当前不要优先做。

## 9. 新 Codex 窗口接手提示词

可以在新窗口直接粘贴：

```text
请读取当前仓库根目录的 CODEX_HANDOFF.md，然后继续 nano-vLLM Qwen3-4B speculative decoding 项目。

当前状态应视为 6.10.1 稳定版本：native nano draft + draft decode CUDA graph + greedy target verification + profiling-only 细分耗时统计。当前最佳 speculative length 是 k9，5 次 repeat 均值约 127.74 tok/s。不要恢复 6.11 target verification CUDA graph，不要恢复 6.11.2 draft cache repair，也不要恢复 6.10.2 draft runner CUDA graph fusion，因为三者都已在云端验证失败或回退。

请先确认本地测试通过，然后根据 CODEX_HANDOFF.md 给出的命令同步到云端 ssh -p 23658 root@connect.cqa1.seetacloud.com，优先跑 k9 repeat 或更长输出长度 benchmark，确认性能稳定在 125+ tok/s。之后再基于 profiling 结果制定下一步优化计划。
```

## 10. 常用运行、测试、提交命令

### 本地测试

```bash
cd /Users/huangs/Documents/Codex/2026-06-30/mac-nano-vllm-scripts-run-qwen3/work/nano-vllm

python3 -m unittest discover -s tests
python3 -m compileall nanovllm bench.py scripts tests
find . -type d -name __pycache__ -prune -exec rm -rf {} +
```

### 同步到云端

```bash
rsync -avz --progress -e "ssh -p 23658" \
  --exclude '__pycache__' \
  --exclude '.git' \
  --exclude '.DS_Store' \
  --exclude 'outputs' \
  /Users/huangs/Documents/Codex/2026-06-30/mac-nano-vllm-scripts-run-qwen3/work/nano-vllm/ \
  root@connect.cqa1.seetacloud.com:/root/nano-vllm/
```

### 云端环境检查

```bash
ssh -p 23658 root@connect.cqa1.seetacloud.com
cd /root/nano-vllm

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0))
try:
    import flash_attn
    print("flash-attn:", flash_attn.__version__)
except Exception as exc:
    print("flash-attn missing:", exc)
PY
```

### 云端依赖安装

```bash
cd /root/nano-vllm
pip install -e .
pip install transformers safetensors sentencepiece tiktoken blobfile tqdm
```

如果 `flash-attn` 缺失，优先安装数据盘里的 wheel：

```bash
pip install /root/autodl-tmp/wheels/flash_attn-*.whl
```

### Baseline benchmark

```bash
python bench.py \
  --model /root/autodl-tmp/huggingface/Qwen3-4B \
  --gpu-memory-utilization 0.75 \
  --num-seqs 1 \
  --max-input-len 512 \
  --max-output-len 256 \
  --temperature 0.01 \
  --out /root/autodl-tmp/nano-vllm-outputs/bench-baseline-single.json
```

### k7 speculative benchmark

```bash
python bench.py \
  --model /root/autodl-tmp/huggingface/Qwen3-4B \
  --enable-engine-speculative \
  --draft-type nano \
  --draft-model /root/autodl-tmp/huggingface/Qwen3-0.6B \
  --gpu-memory-utilization 0.75 \
  --num-speculative-tokens 7 \
  --num-seqs 1 \
  --max-input-len 512 \
  --max-output-len 256 \
  --temperature 0.01 \
  --speculative-greedy-temperature 0.011 \
  --out /root/autodl-tmp/nano-vllm-outputs/bench-nano-draft-k7-6.10-restore.json
```

### k sweep

```bash
for k in 3 5 7; do
  python bench.py \
    --model /root/autodl-tmp/huggingface/Qwen3-4B \
    --enable-engine-speculative \
    --draft-type nano \
    --draft-model /root/autodl-tmp/huggingface/Qwen3-0.6B \
    --gpu-memory-utilization 0.75 \
    --num-speculative-tokens $k \
    --num-seqs 1 \
    --max-input-len 512 \
    --max-output-len 256 \
    --temperature 0.01 \
    --speculative-greedy-temperature 0.011 \
    --out /root/autodl-tmp/nano-vllm-outputs/bench-nano-draft-k${k}-6.10-restore.json
done
```

### 汇总

```bash
python scripts/summarize_benchmarks.py \
  /root/autodl-tmp/nano-vllm-outputs/bench-baseline-single.json \
  /root/autodl-tmp/nano-vllm-outputs/bench-nano-draft-k3-6.10-restore.json \
  /root/autodl-tmp/nano-vllm-outputs/bench-nano-draft-k5-6.10-restore.json \
  /root/autodl-tmp/nano-vllm-outputs/bench-nano-draft-k7-6.10-restore.json \
  --out /root/autodl-tmp/nano-vllm-outputs/bench-nano-draft-sweep-6.10-restore.md

cat /root/autodl-tmp/nano-vllm-outputs/bench-nano-draft-sweep-6.10-restore.md
```

### Git 状态

```bash
git status --short
git diff --stat
git diff
```

### 提交

```bash
git add README.md bench.py pyproject.toml nanovllm scripts tests CODEX_HANDOFF.md
git commit -m "Add Qwen3 speculative decoding benchmark path"
```
