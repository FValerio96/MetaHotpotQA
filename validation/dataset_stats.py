"""
dataset_stats.py — Statistics for the MetaHotpotQA dataset.

Computes distributions over difficulty, reasoning_type, and S0 connectivity
(connected vs disconnected seed nodes) on one or more splits.

Usage:
    python validation/dataset_stats.py
    python validation/dataset_stats.py --splits published_splits
    python validation/dataset_stats.py --splits published_splits --split test
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

DEFAULT_SPLITS_DIR = "published_splits"


def stats_for_split(path: Path) -> dict:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    total = len(records)
    by_difficulty = defaultdict(int)
    by_reasoning = defaultdict(int)
    s0_connected = 0
    s0_disconnected = 0

    for r in records:
        diff = r.get("difficulty", "unknown")
        rtype = r.get("reasoning_type", "unknown")
        by_difficulty[diff] += 1
        by_reasoning[rtype] += 1

        if diff == "entity_selection":
            if r.get("supporting_paths"):
                s0_connected += 1
            else:
                s0_disconnected += 1

    s0_total = s0_connected + s0_disconnected
    return {
        "total": total,
        "by_difficulty": dict(by_difficulty),
        "by_reasoning_type": dict(by_reasoning),
        "entity_selection": {
            "total": s0_total,
            "connected": s0_connected,
            "disconnected": s0_disconnected,
            "connected_pct": round(100 * s0_connected / s0_total, 1) if s0_total else 0,
            "disconnected_pct": round(100 * s0_disconnected / s0_total, 1) if s0_total else 0,
        },
    }


def print_stats(name: str, s: dict) -> None:
    total = s["total"]
    print(f"\n{'='*55}")
    print(f"Split: {name}  ({total} questions)")
    print(f"{'='*55}")

    print("\nDifficulty distribution:")
    for diff, count in sorted(s["by_difficulty"].items()):
        print(f"  {diff:<25} {count:>6}  ({100*count/total:5.1f}%)")

    print("\nReasoning type distribution:")
    for rtype, count in sorted(s["by_reasoning_type"].items()):
        print(f"  {rtype:<25} {count:>6}  ({100*count/total:5.1f}%)")

    es = s["entity_selection"]
    if es["total"]:
        print(f"\nEntity selection — seed node connectivity:")
        print(f"  Total S0:      {es['total']:>6}")
        print(f"  Connected:     {es['connected']:>6}  ({es['connected_pct']:.1f}%)")
        print(f"  Disconnected:  {es['disconnected']:>6}  ({es['disconnected_pct']:.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-split statistics of the MetaHotpotQA dataset"
    )
    parser.add_argument(
        "--splits", default=DEFAULT_SPLITS_DIR,
        help=f"Directory of the JSONL splits. Default: {DEFAULT_SPLITS_DIR}"
    )
    parser.add_argument(
        "--split", default=None,
        help="Process only this split (e.g. train, dev, test)"
    )
    args = parser.parse_args()

    splits_dir = Path(args.splits)
    split_files = (
        [splits_dir / f"{args.split}.jsonl"] if args.split
        else sorted(splits_dir.glob("*.jsonl"))
    )

    totals = defaultdict(int)
    total_s0_connected = 0
    total_s0_disconnected = 0
    total_by_diff = defaultdict(int)

    for path in split_files:
        if not path.exists():
            print(f"File not found: {path}")
            continue
        s = stats_for_split(path)
        print_stats(path.stem, s)
        totals["total"] += s["total"]
        total_s0_connected += s["entity_selection"]["connected"]
        total_s0_disconnected += s["entity_selection"]["disconnected"]
        for diff, count in s["by_difficulty"].items():
            total_by_diff[diff] += count

    if len(split_files) > 1:
        grand_total = totals["total"]
        s0_total = total_s0_connected + total_s0_disconnected
        print(f"\n{'='*55}")
        print(f"TOTAL  ({grand_total} questions)")
        print(f"{'='*55}")
        print("\nDifficulty distribution:")
        for diff, count in sorted(total_by_diff.items()):
            print(f"  {diff:<25} {count:>6}  ({100*count/grand_total:5.1f}%)")
        if s0_total:
            print(f"\nEntity selection — seed node connectivity:")
            print(f"  Total S0:      {s0_total:>6}")
            print(f"  Connected:     {total_s0_connected:>6}  ({100*total_s0_connected/s0_total:.1f}%)")
            print(f"  Disconnected:  {total_s0_disconnected:>6}  ({100*total_s0_disconnected/s0_total:.1f}%)")


if __name__ == "__main__":
    main()
