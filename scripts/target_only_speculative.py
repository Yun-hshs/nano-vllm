from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanovllm.speculative import DSparkSpeculativeConfig, run_ngram_target_only_experiment


def parse_token_trace(value: str) -> list[int]:
    path = Path(value)
    if path.exists():
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
        if isinstance(data, dict):
            data = data["token_ids"]
        return [int(token_id) for token_id in data]
    return [int(token_id.strip()) for token_id in value.split(",") if token_id.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Run target-only speculative verification on a token trace.")
    parser.add_argument("--token-trace", required=True, help="Comma-separated token ids or a JSON file/list.")
    parser.add_argument("--prefix-len", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--ngram-size", type=int, default=4)
    parser.add_argument("--num-speculative-tokens", type=int, default=DSparkSpeculativeConfig().num_speculative_tokens)
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument("--out", help="Optional JSON output path.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = DSparkSpeculativeConfig(
        num_speculative_tokens=args.num_speculative_tokens,
        confidence_threshold=args.confidence_threshold,
    )
    result = run_ngram_target_only_experiment(
        token_trace=parse_token_trace(args.token_trace),
        prefix_len=args.prefix_len,
        max_tokens=args.max_tokens,
        ngram_size=args.ngram_size,
        config=config,
    )
    data = {
        "output_token_ids": result.output_token_ids,
        "proposed_tokens": result.proposed_tokens,
        "accepted_tokens": result.accepted_tokens,
        "target_forwards": result.target_forwards,
        "elapsed_seconds": result.elapsed_seconds,
        "tokens_per_second": result.tokens_per_second,
        "acceptance_lengths": result.acceptance_lengths,
        "mean_acceptance_length": result.mean_acceptance_length,
    }
    text = json.dumps(data, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
