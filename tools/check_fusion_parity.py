#!/usr/bin/env python3
"""End-to-end token/logit parity for the fused Triton operators.

Runs the real Qwen3 model twice in separate processes (fusion off = the
torch.compile chain, fusion on = the single-pass Triton kernels) and compares
them on *identical inputs*:

  * run A generates greedily with fusion off and dumps its token stream and the
    per-step logits;
  * run B is forced to consume run A's tokens at every step (teacher forcing) so
    both runs see exactly the same context, and dumps its own logits;
  * the comparison is per-step logit cosine plus the fraction of steps where
    run B's argmax equals run A's token -- i.e. how often fusion on would have
    taken the greedy branch that fusion off took.

    python tools/check_fusion_parity.py --model ~/huggingface/Qwen3-0.6B
    python tools/check_fusion_parity.py --model ... --eager --max-tokens 32
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

import torch

# Allow running this file directly (`python tools/check_fusion_parity.py`) even
# when another nano-vllm copy is installed in site-packages.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _build_prompts(num_prompts, prompt_len, vocab_size, seed):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, vocab_size, (num_prompts, prompt_len), generator=g)
    return ids.tolist()


def run_worker(args):
    import torch.nn.functional as F

    from nanovllm import LLM, SamplingParams
    from nanovllm.layers import fused_ops

    ref = None
    if args.forced_tokens:
        # Flat per-run_model-call greedy stream, in execution order (decode batch,
        # then prefill chunk for every step).
        ref_calls = torch.load(args.forced_tokens, weights_only=False)["tokens"]
        ref = [t for call in ref_calls for t in call]

    llm = LLM(
        args.model,
        enforce_eager=args.eager,
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        triton_fusion=bool(args.fusion),
    )
    assert fused_ops.fused_enabled() == bool(args.fusion), "fusion flag did not take effect"

    runner = llm.model_runner
    captured = {"logits": [], "tokens": []}

    orig_run_model = runner.run_model

    @torch.inference_mode()
    def run_model(input_ids, positions, is_prefill):
        logits = orig_run_model(input_ids, positions, is_prefill)
        captured["logits"].append(logits.detach().float().cpu())
        return logits

    orig_run = runner.run
    cursor = {"i": 0}

    @torch.inference_mode()
    def run(decode_seqs, prefill_seqs):
        before = len(captured["logits"])
        decode_token_ids, prefill_token_ids = orig_run(decode_seqs, prefill_seqs)
        # One run_model call per non-empty batch, in execution order: the decode
        # graph batch first, then the eager chunked-prefill batch.
        batches = [b for b in (decode_seqs, prefill_seqs) if b]
        sampled = []
        for batch, logits in zip(batches, captured["logits"][before:]):
            greedy = logits.argmax(dim=-1).tolist()
            captured["tokens"].append(greedy)
            if ref is not None:
                # Teacher forcing: replay run A's token stream so the next step's
                # inputs are identical on both sides.
                i = cursor["i"]
                greedy = ref[i:i + len(batch)]
                cursor["i"] = i + len(batch)
            sampled.append(greedy)
        if ref is None:
            return decode_token_ids, prefill_token_ids
        it = iter(sampled)
        return (next(it) if decode_seqs else []), (next(it) if prefill_seqs else [])

    runner.run_model = run_model
    runner.run = run

    prompts = _build_prompts(args.num_prompts, args.prompt_len, args.vocab_size, args.seed)
    sp = SamplingParams(temperature=1.0, ignore_eos=True, max_tokens=args.max_tokens)
    for p in prompts:
        llm.add_request(list(p), sp)
    while not llm.is_finished():
        llm.step()

    logits = captured["logits"]
    tokens = captured["tokens"]
    # cosine of each step's logits against the previous step's (sanity: the tool
    # must produce a non-degenerate stream, a stuck model would cos ~1)
    torch.save({"logits": logits, "tokens": tokens}, args.out)
    cos_prev = []
    for i in range(1, len(logits)):
        cos_prev.append(
            float(F.cosine_similarity(logits[i].flatten(), logits[i - 1].flatten(), dim=0))
        )
    print("META " + json.dumps({
        "steps": len(logits),
        "mean_step_cosine_vs_prev": sum(cos_prev) / len(cos_prev) if cos_prev else None,
    }))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--num-prompts", type=int, default=3)
    parser.add_argument("--prompt-len", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--eager", action="store_true", help="enforce eager (no CUDA graph)")
    # internal worker plumbing
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--fusion", type=int, default=0)
    parser.add_argument("--forced-tokens", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.worker:
        run_worker(args)
        return

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tmp = tempfile.mkdtemp(prefix="nanovllm-fusion-parity-")
    ref_path = os.path.join(tmp, "ref.pt")
    fused_path = os.path.join(tmp, "fused.pt")

    def spawn(extra):
        cmd = [
            sys.executable, os.path.join(repo, "tools", "check_fusion_parity.py"),
            "--worker", "--model", args.model,
            "--num-prompts", str(args.num_prompts), "--prompt-len", str(args.prompt_len),
            "--max-tokens", str(args.max_tokens), "--vocab-size", str(args.vocab_size),
            "--seed", str(args.seed), "--max-model-len", str(args.max_model_len),
            *(["--eager"] if args.eager else []),
            *extra,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=repo)
        if proc.returncode != 0:
            raise SystemExit("worker failed:\n" + proc.stdout[-3000:] + proc.stderr[-3000:])
        meta = [l for l in proc.stdout.splitlines() if l.startswith("META ")]
        assert meta, "worker produced no META line:\n" + proc.stdout[-2000:]
        return json.loads(meta[-1][len("META "):])

    ref_meta = spawn(["--fusion", "0", "--out", ref_path])
    fused_meta = spawn(["--fusion", "1", "--forced-tokens", ref_path, "--out", fused_path])

    ref = torch.load(ref_path, weights_only=False)
    fused = torch.load(fused_path, weights_only=False)
    assert len(ref["logits"]) == len(fused["logits"]), (
        f"step count diverged: {len(ref['logits'])} vs {len(fused['logits'])}"
    )

    cosines, matches, total = [], 0, 0
    for ref_l, fused_l, ref_tok in zip(ref["logits"], fused["logits"], ref["tokens"]):
        assert ref_l.shape == fused_l.shape, (ref_l.shape, fused_l.shape)
        cosines.append(float(torch.nn.functional.cosine_similarity(
            ref_l.flatten(), fused_l.flatten(), dim=0)))
        pred = fused_l.argmax(dim=-1).tolist()
        matches += sum(int(a == b) for a, b in zip(pred, ref_tok))
        total += len(ref_tok)

    summary = {
        "steps": len(cosines),
        "tokens_compared": total,
        "token_match_rate": matches / max(total, 1),
        "worst_logit_cosine": min(cosines),
        "mean_logit_cosine": sum(cosines) / len(cosines),
        "max_abs_logit_diff": max(float((a - b).abs().max()) for a, b in zip(ref["logits"], fused["logits"])),
        "ref_stream_step_cosine": ref_meta["mean_step_cosine_vs_prev"],
        "fused_stream_step_cosine": fused_meta["mean_step_cosine_vs_prev"],
    }
    print("SUMMARY " + json.dumps(summary))
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:<26} {v:.6f}")
        else:
            print(f"  {k:<26} {v}")


if __name__ == "__main__":
    main()
