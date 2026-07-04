from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanovllm.speculative import benchmark_result_from_dict, format_benchmark_comparison


def parse_args():
    parser = argparse.ArgumentParser(description="Compare baseline and speculative benchmark JSON files.")
    parser.add_argument("baseline_json")
    parser.add_argument("speculative_json")
    parser.add_argument("--out", help="Optional Markdown output path.")
    return parser.parse_args()


def main():
    args = parse_args()
    baseline = benchmark_result_from_dict(json.loads(Path(args.baseline_json).read_text()))
    speculative = benchmark_result_from_dict(json.loads(Path(args.speculative_json).read_text()))
    report = format_benchmark_comparison(baseline, speculative)
    if args.out:
        Path(args.out).write_text(report + "\n", encoding="utf-8")
    else:
        print(report)


if __name__ == "__main__":
    main()
