from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanovllm.speculative import BenchmarkResult, benchmark_result_from_dict


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize multiple benchmark JSON files.")
    parser.add_argument("baseline_json")
    parser.add_argument("benchmark_json", nargs="+")
    parser.add_argument("--out", help="Optional Markdown output path.")
    return parser.parse_args()


def label_for(path: Path, result: BenchmarkResult) -> str:
    stem = path.stem
    for marker in ("k3", "k5", "k7"):
        if marker in stem:
            return marker
    return result.mode


def format_summary(baseline: BenchmarkResult, rows: list[tuple[str, BenchmarkResult]]) -> str:
    lines = [
        "# Benchmark Sweep Summary",
        "",
        "| label | throughput (tok/s) | speedup | acceptance | mean accepted | draft forwards | draft rebuilds | cached steps | graph enabled | graph replays | target forwards | greedy forwards | draft propose ms | draft prefill ms | draft cache extend ms | draft graph ms | draft eager ms | draft select ms | target verify ms | target forward ms | target argmax ms | target compare ms | append ms | step ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, result in rows:
        acceptance = "-" if result.acceptance_rate is None else f"{result.acceptance_rate * 100:.2f}%"
        lines.append(
            f"| {label} | "
            f"{result.tokens_per_second:.2f} | "
            f"{result.tokens_per_second / baseline.tokens_per_second:.2f}x | "
            f"{acceptance} | "
            f"{result.mean_acceptance_length:.2f} | "
            f"{result.draft_forwards} | "
            f"{result.draft_rebuilds} | "
            f"{result.draft_cached_steps} | "
            f"{'yes' if result.draft_graph_enabled else 'no'} | "
            f"{result.draft_graph_replays} | "
            f"{result.target_forwards} | "
            f"{result.target_greedy_forwards} | "
            f"{result.draft_propose_ms:.2f} | "
            f"{result.draft_prefill_ms:.2f} | "
            f"{result.draft_cache_extend_ms:.2f} | "
            f"{result.draft_decode_graph_ms:.2f} | "
            f"{result.draft_decode_eager_ms:.2f} | "
            f"{result.draft_token_select_ms:.2f} | "
            f"{result.target_verify_ms:.2f} | "
            f"{result.target_forward_ms:.2f} | "
            f"{result.target_argmax_ms:.2f} | "
            f"{result.target_compare_ms:.2f} | "
            f"{result.append_tokens_ms:.2f} | "
            f"{result.speculative_step_ms:.2f} |"
        )
    return "\n".join(lines)


def main():
    args = parse_args()
    baseline_path = Path(args.baseline_json)
    baseline = benchmark_result_from_dict(json.loads(baseline_path.read_text()))
    rows = []
    for benchmark_json in args.benchmark_json:
        path = Path(benchmark_json)
        result = benchmark_result_from_dict(json.loads(path.read_text()))
        rows.append((label_for(path, result), result))
    report = format_summary(baseline, rows)
    if args.out:
        Path(args.out).write_text(report + "\n", encoding="utf-8")
    else:
        print(report)


if __name__ == "__main__":
    main()
