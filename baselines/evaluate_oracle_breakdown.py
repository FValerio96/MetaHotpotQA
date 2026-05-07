"""
evaluate_oracle_breakdown.py — Iterates over all preds_*.jsonl files in a
results directory and produces evaluation with the oracle_full / oracle_partial
breakdown.

context_type definition (derived from the predictions x split join on id):
  oracle_full:    traversal
                  entity_selection with non-empty supporting_paths
  oracle_partial: property_comparison
                  entity_selection without supporting_paths

Output: <results_dir>/evals_context/<stem>.json for each preds file.

Usage:
    python -m baselines.evaluate_oracle_breakdown
    python -m baselines.evaluate_oracle_breakdown --results_dir baselines/results/oracle
    python -m baselines.evaluate_oracle_breakdown --split published_splits/test.jsonl
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from baselines.evaluate import (
    exact_match,
    hits_at_1,
    load_predictions,
    load_split,
    token_f1,
)

_PROJECT_ROOT  = Path(__file__).resolve().parent.parent
DEFAULT_SPLIT   = str(_PROJECT_ROOT / "published_splits" / "test.jsonl")
DEFAULT_RESULTS = str(_PROJECT_ROOT / "baselines" / "results" / "oracle")


# ---------------------------------------------------------------------------
# context_type helper
# ---------------------------------------------------------------------------

def _context_type(record: dict) -> str:
    difficulty = record.get("difficulty", "")
    has_path   = bool(record.get("supporting_paths"))
    if difficulty == "traversal":
        return "oracle_full"
    if difficulty == "entity_selection":
        return "oracle_full" if has_path else "oracle_partial"
    if difficulty == "property_comparison":
        return "oracle_partial"
    return "oracle_partial"


# ---------------------------------------------------------------------------
# Evaluation with context_type breakdown
# ---------------------------------------------------------------------------

def evaluate_with_context(predictions: dict[str, dict],
                          records: list[dict]) -> dict:
    global_em, global_f1 = [], []
    by_difficulty:    dict[str, dict] = defaultdict(lambda: {"em": [], "f1": []})
    by_context:       dict[str, dict] = defaultdict(lambda: {"em": [], "f1": []})
    by_reasoning:     dict[str, dict] = defaultdict(lambda: {"em": [], "f1": []})
    missing = 0

    for record in records:
        rid = record["id"]
        if rid not in predictions:
            missing += 1
            continue

        pred         = predictions[rid]
        pred_answer  = pred.get("pred_answer", "")
        gold_aliases = record.get("answer_aliases") or [record["answer"]]
        difficulty   = record.get("difficulty", "unknown")
        rtype        = record.get("reasoning_type", "unknown")
        ctype        = _context_type(record)

        em = exact_match(pred_answer, gold_aliases)
        f1 = token_f1(pred_answer, gold_aliases)

        global_em.append(em)
        global_f1.append(f1)
        by_difficulty[difficulty]["em"].append(em)
        by_difficulty[difficulty]["f1"].append(f1)
        by_context[ctype]["em"].append(em)
        by_context[ctype]["f1"].append(f1)
        by_reasoning[rtype]["em"].append(em)
        by_reasoning[rtype]["f1"].append(f1)

    def avg(lst):
        return round(sum(lst) / len(lst) * 100, 1) if lst else 0.0

    def summarise(d):
        return {k: {"em": avg(v["em"]), "f1": avg(v["f1"]), "n": len(v["em"])}
                for k, v in sorted(d.items())}

    return {
        "n_evaluated":             len(global_em),
        "n_missing_predictions":   missing,
        "global":                  {"em": avg(global_em), "f1": avg(global_f1)},
        "by_context_type":         summarise(by_context),
        "by_difficulty":           summarise(by_difficulty),
        "by_reasoning_type":       summarise(by_reasoning),
    }


# ---------------------------------------------------------------------------
# Print
# ---------------------------------------------------------------------------

def print_results(model_name: str, results: dict) -> None:
    g = results["global"]
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  MODEL: {model_name}")
    print(f"  N evaluated : {results['n_evaluated']}")
    if results["n_missing_predictions"]:
        print(f"  Missing preds: {results['n_missing_predictions']}")
    print(sep)
    print(f"  GLOBAL  EM: {g['em']:.1f}%  F1: {g['f1']:.1f}%")
    print(sep)

    print("\n  By context type:")
    for ctype, m in results["by_context_type"].items():
        print(f"    {ctype:<20} n={m['n']:<5} EM:{m['em']:.1f}%  F1:{m['f1']:.1f}%")

    print("\n  By difficulty:")
    for diff, m in results["by_difficulty"].items():
        print(f"    {diff:<25} n={m['n']:<5} EM:{m['em']:.1f}%  F1:{m['f1']:.1f}%")

    print("\n  By reasoning type:")
    for rtype, m in results["by_reasoning_type"].items():
        print(f"    {rtype:<25} n={m['n']:<5} EM:{m['em']:.1f}%  F1:{m['f1']:.1f}%")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate all models in a results directory with the context_type breakdown"
    )
    parser.add_argument("--results_dir", default=DEFAULT_RESULTS,
                        help=f"Directory containing the preds_*.jsonl files. Default: {DEFAULT_RESULTS}")
    parser.add_argument("--split", default=DEFAULT_SPLIT,
                        help=f"JSONL file of the split. Default: {DEFAULT_SPLIT}")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir  = results_dir / "evals_context"
    output_dir.mkdir(exist_ok=True)

    records    = load_split(Path(args.split))
    pred_files = sorted(results_dir.glob("preds_*.jsonl"))

    if not pred_files:
        print(f"No preds_*.jsonl files found in {results_dir}")
        return

    for pred_path in pred_files:
        model_name  = pred_path.stem.removeprefix("preds_")
        predictions = load_predictions(pred_path)
        results     = evaluate_with_context(predictions, records)

        print_results(model_name, results)

        out_path = output_dir / f"{pred_path.stem}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"model": model_name, **results}, f, indent=2, ensure_ascii=False)
        print(f"  Saved to {out_path}")


if __name__ == "__main__":
    main()
