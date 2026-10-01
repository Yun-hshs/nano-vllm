#!/usr/bin/env python3
"""Logits-level alignment harness: FP16 vs FP8 E4M3 KV cache.

Each precision runs in its own subprocess, because a previous LLM's KV cache
and CUDA peak-memory stats leak into a second `ModelRunner` built in the same
process (and would make its KV allocation fail).

To make the comparison meaningful the sampler is replaced by a deterministic
greedy recorder: the FP16 run records the token it picks at every step, and the
FP8 run is forced to consume exactly that token stream. Both runs therefore
process identical inputs at every step, so their logits are directly comparable
and no step is lost to sampling divergence (RNG is not involved at all).

Two phases are run so both FP8 paths are covered:
  phase 1  long prompts fill the paged cache (plain prefill + decode kernel)
  phase 2  the same prompts plus a suffix hit the prefix cache, which forces
           the gather + dequantize prefill path

    python tools/logits_align.py --model /root/huggingface/Qwen3-0.6B
    python tools/logits_align.py --model ... --k-margin 2.0 --calib-tokens 2048
"""

import argparse
import json
import os
import subprocess
import sys

NUM_PROMPTS = 4
PROMPT_LEN = 600
SUFFIX_LEN = 60


def make_prompts():
    prompts, suffixed = [], []
    for i in range(NUM_PROMPTS):
        base = 100 + i * 1000
        prompt = list(range(base, base + PROMPT_LEN))
        prompts.append(prompt)
        suffixed.append(prompt + list(range(base + 5000, base + 5000 + SUFFIX_LEN)))
    return prompts, suffixed


class GreedyRecorder:
    """Drop-in sampler: records greedy tokens, or replays a forced stream."""

    def __init__(self, forced=None):
        self.forced = forced
        self.recorded = []
        self.index = 0

    def __call__(self, logits, temperatures):
        import torch
        if self.forced is not None:
            tokens = torch.tensor(self.forced[self.index], dtype=torch.long, device=logits.device)
        else:
            tokens = logits.argmax(dim=-1)
        self.recorded.append(tokens.tolist())
        self.index += 1
        return tokens


def run_one(args):
    import torch
    import nanovllm.layers.attention as A
    from nanovllm import LLM, SamplingParams

    llm = LLM(
        args.model,
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
        fp8_kvcache=bool(args.fp8),
        fp8_kv_calib_tokens=args.calib_tokens,
        fp8_kv_k_margin=args.k_margin,
    )

    forced = None
    if args.force_tokens:
        forced = torch.load(args.force_tokens, weights_only=False)["recorded"]
    recorder = GreedyRecorder(forced=forced)
    llm.model_runner.sampler = recorder

    captured = []
    gather_calls = {"n": 0}
    original_gather = A.gather_dequant_kvcache

    def gather_counted(*a, **kw):
        gather_calls["n"] += 1
        return original_gather(*a, **kw)

    A.gather_dequant_kvcache = gather_counted

    original_run_model = llm.model_runner.run_model

    def patched_run_model(input_ids, positions, is_prefill):
        logits = original_run_model(input_ids, positions, is_prefill)
        captured.append(logits.detach().float().cpu())
        return logits

    llm.model_runner.run_model = patched_run_model

    prompts, suffixed = make_prompts()
    # temperature/max_tokens are irrelevant to the greedy recorder, but keep them
    # consistent so both runs schedule identically
    sampling = [SamplingParams(temperature=1.0, ignore_eos=True, max_tokens=args.max_tokens)] * NUM_PROMPTS

    llm.generate(prompts, sampling, use_tqdm=False)
    boundary = len(captured)
    llm.generate(suffixed, sampling, use_tqdm=False)

    torch.save(
        {
            "logits": captured,
            "boundary": boundary,
            "gather_calls": gather_calls["n"],
            "recorded": recorder.recorded,
        },
        args.out,
    )


def compare(path_a, path_b):
    import torch

    a = torch.load(path_a, weights_only=False)
    b = torch.load(path_b, weights_only=False)
    la, lb = a["logits"], b["logits"]
    boundary = a["boundary"]
    print(f"steps: fp16={len(la)} fp8={len(lb)}  phase1 steps={boundary}  fp8 gather calls={b['gather_calls']}")
    assert len(la) == len(lb), (len(la), len(lb))
    print(f"token streams identical (forced): {a['recorded'] == b['recorded']}")

    summaries = {}
    for phase, (lo, hi) in (("phase1", (0, boundary)), ("phase2", (boundary, len(la)))):
        worst_cos, worst_abs, top1_hits, top1_total = 1.0, 0.0, 0, 0
        prefill_cos = None
        print(f"--- {phase} ---")
        for step in range(lo, hi):
            x, y = la[step], lb[step]
            cos = torch.nn.functional.cosine_similarity(x.flatten(), y.flatten(), dim=0).item()
            max_abs = (x - y).abs().max().item()
            top1 = (x.argmax(-1) == y.argmax(-1)).float().mean().item()
            kind = "prefill" if step == lo else "decode"
            if kind == "prefill":
                prefill_cos = cos
            else:
                worst_cos = min(worst_cos, cos)
                worst_abs = max(worst_abs, max_abs)
                top1_hits += int(round(top1 * x.shape[0]))
                top1_total += x.shape[0]
            print(f"  step {step:2d} ({kind:7s}): cosine={cos:.6f}  max|dlogit|={max_abs:.4f}  top1={top1:.3f}")
        summaries[phase] = {
            "prefill_cosine": prefill_cos,
            "scored_decode_steps": top1_total // max(1, x.shape[0]),
            "worst_decode_cosine": worst_cos,
            "worst_decode_max_abs": worst_abs,
            "decode_top1_agreement": (top1_hits / top1_total) if top1_total else 1.0,
        }
    summary = {"gather_calls": b["gather_calls"], **summaries}
    print("SUMMARY " + json.dumps(summary))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.path.expanduser("~/huggingface/Qwen3-0.6B"))
    parser.add_argument("--mode", choices=["all", "run", "compare"], default="all")
    parser.add_argument("--fp8", type=int, choices=[0, 1], default=0)
    parser.add_argument("--out", default="/tmp/logits.pt")
    parser.add_argument("--a", default="/tmp/logits_fp16.pt")
    parser.add_argument("--b", default="/tmp/logits_fp8.pt")
    parser.add_argument("--force-tokens", default=None)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--calib-tokens", type=int, default=512)
    parser.add_argument("--k-margin", type=float, default=1.5)
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()

    if args.mode == "run":
        run_one(args)
        return

    if args.mode == "all":
        common = [
            "--model", args.model,
            "--max-model-len", str(args.max_model_len),
            "--max-tokens", str(args.max_tokens),
            "--calib-tokens", str(args.calib_tokens),
        ]
        if args.enforce_eager:
            common.append("--enforce-eager")
        subprocess.run([sys.executable, __file__, "--mode", "run", "--fp8", "0", "--out", args.a] + common, check=True)
        subprocess.run([sys.executable, __file__, "--mode", "run", "--fp8", "1", "--out", args.b,
                        "--force-tokens", args.a, "--k-margin", str(args.k_margin)] + common, check=True)
        compare(args.a, args.b)
        return

    compare(args.a, args.b)


if __name__ == "__main__":
    main()
