"""
evaluate.py — Shared evaluation module for the MetaHotpotQA baselines.

Metrics:
  - Exact Match (EM): 1 if the prediction matches at least one gold alias
                      after normalization (lowercase, strip punctuation).
  - Token-level F1:   max F1 between the prediction and all gold aliases.
  - Hits@1 on QID:    1 if the predicted QID matches answer_node_qid.
                      Used by KG-based systems that predict entities, not strings.

Each metric is reported globally and broken down by:
  - difficulty: entity_selection / direct_traversal / waypoint_traversal /
                property_comparison
  - reasoning_type: bridge / comparison
  - ontology: ont_1_movie / ont_2_music / ... (multi-label: one question
              may appear under multiple domains)

Standalone usage:
    python evaluate.py --predictions PATH --split PATH [--output PATH]

The predictions file must be JSONL with at least the fields:
    {"id": "...", "pred_answer": "...", "pred_qid": "..."}
    (pred_qid is optional, used only if present)
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    """Lowercase, remove articles, strip punctuation, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Per-prediction metrics
# ---------------------------------------------------------------------------

def exact_match(prediction: str, gold_aliases: list[str]) -> float:
    """1.0 if the normalized prediction matches at least one gold alias."""
    pred_norm = normalize(prediction)
    return float(any(pred_norm == normalize(g) for g in gold_aliases))


def token_f1(prediction: str, gold_aliases: list[str]) -> float:
    """Maximum token-level F1 between the prediction and all gold aliases."""
    pred_tokens = normalize(prediction).split()
    if not pred_tokens:
        return 0.0
    best = 0.0
    for gold in gold_aliases:
        gold_tokens = normalize(gold).split()
        if not gold_tokens:
            continue
        common = len(set(pred_tokens) & set(gold_tokens))
        if common == 0:
            continue
        p = common / len(pred_tokens)
        r = common / len(gold_tokens)
        f1 = 2 * p * r / (p + r)
        best = max(best, f1)
    return best


def hits_at_1(pred_qid: str | None, gold_qid: str) -> float:
    """1.0 if the predicted QID matches the gold QID. Requires pred_qid."""
    if pred_qid is None:
        return 0.0
    return float(pred_qid.strip() == gold_qid.strip())


# ---------------------------------------------------------------------------
# Evaluation over a full split
# ---------------------------------------------------------------------------

def evaluate(predictions: dict[str, dict], records: list[dict]) -> dict:
    """
    Compute all metrics over a set of predictions.

    Args:
        predictions: dict id -> {"pred_answer": str, "pred_qid": str|None}
        records:     list of records from the split file (with gold fields)

    Returns:
        dict with global metrics and breakdowns by difficulty and reasoning_type.
    """
    global_em, global_f1, global_h1 = [], [], []
    by_difficulty: dict[str, dict] = defaultdict(lambda: {"em": [], "f1": [], "h1": []})
    by_type: dict[str, dict] = defaultdict(lambda: {"em": [], "f1": [], "h1": []})
    by_ontology: dict[str, dict] = defaultdict(lambda: {"em": [], "f1": [], "h1": []})
    missing = 0

    for record in records:
        qid = record["id"]
        if qid not in predictions:
            missing += 1
            continue

        pred = predictions[qid]
        gold_aliases = record.get("answer_aliases") or [record["answer"]]
        gold_qid = record.get("answer_node_qid", "")
        difficulty = record.get("difficulty", "unknown")
        rtype = record.get("reasoning_type", "unknown")
        ontologies = [o["ont_id"] for o in record.get("matched_ontologies", [])]

        pred_answer = pred.get("pred_answer", "")
        pred_qid = pred.get("pred_qid")

        em = exact_match(pred_answer, gold_aliases)
        f1 = token_f1(pred_answer, gold_aliases)
        h1 = hits_at_1(pred_qid, gold_qid)

        global_em.append(em)
        global_f1.append(f1)
        global_h1.append(h1)

        by_difficulty[difficulty]["em"].append(em)
        by_difficulty[difficulty]["f1"].append(f1)
        by_difficulty[difficulty]["h1"].append(h1)

        by_type[rtype]["em"].append(em)
        by_type[rtype]["f1"].append(f1)
        by_type[rtype]["h1"].append(h1)

        for ont in ontologies:
            by_ontology[ont]["em"].append(em)
            by_ontology[ont]["f1"].append(f1)
            by_ontology[ont]["h1"].append(h1)

    def avg(lst):
        return round(sum(lst) / len(lst) * 100, 1) if lst else 0.0


    has_qid = any(p.get("pred_qid") for p in predictions.values())

    def summarise_with_qid(d):
        return {k: {"em": avg(v["em"]), "f1": avg(v["f1"]),
                    "h1": avg(v["h1"]) if has_qid else None, "n": len(v["em"])}
                for k, v in sorted(d.items())}

    return {
        "n_evaluated": len(global_em),
        "n_missing_predictions": missing,
        "global": {
            "em":  avg(global_em),
            "f1":  avg(global_f1),
            "h1":  avg(global_h1) if has_qid else None,
        },
        "by_difficulty":     summarise_with_qid(by_difficulty),
        "by_reasoning_type": summarise_with_qid(by_type),
        "by_ontology":       summarise_with_qid(by_ontology),
    }


def load_predictions(path: Path) -> dict[str, dict]:
    preds = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            preds[r["id"]] = r
    return preds


def load_split(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def print_results(results: dict) -> None:
    g = results["global"]
    print(f"\n{'='*55}")
    print(f"  N evaluated : {results['n_evaluated']}")
    if results["n_missing_predictions"]:
        print(f"  Missing preds: {results['n_missing_predictions']}")
    print(f"{'='*55}")
    h1_str = f"  Hits@1 : {g['h1']:.1f}%" if g["h1"] is not None else ""
    print(f"  GLOBAL  EM: {g['em']:.1f}%  F1: {g['f1']:.1f}%{h1_str}")
    print(f"{'='*55}")

    print("\n  By difficulty:")
    for diff, m in results["by_difficulty"].items():
        h1_str = f"  H@1:{m['h1']:.1f}%" if m["h1"] is not None else ""
        print(f"    {diff:<25} n={m['n']:<5} EM:{m['em']:.1f}%  F1:{m['f1']:.1f}%{h1_str}")

    print("\n  By reasoning type:")
    for rtype, m in results["by_reasoning_type"].items():
        h1_str = f"  H@1:{m['h1']:.1f}%" if m["h1"] is not None else ""
        print(f"    {rtype:<25} n={m['n']:<5} EM:{m['em']:.1f}%  F1:{m['f1']:.1f}%{h1_str}")

    if results.get("by_ontology"):
        print("\n  By ontology (multi-label):")
        for ont, m in results["by_ontology"].items():
            h1_str = f"  H@1:{m['h1']:.1f}%" if m["h1"] is not None else ""
            print(f"    {ont:<25} n={m['n']:<5} EM:{m['em']:.1f}%  F1:{m['f1']:.1f}%{h1_str}")
    print()


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline predictions on MetaHotpotQA")
    parser.add_argument("--predictions", required=True, help="JSONL file with predictions")
    parser.add_argument("--split",       required=True, help="JSONL file of the split (test/dev)")
    parser.add_argument("--output",      help="Save results as JSON (optional)")
    args = parser.parse_args()

    predictions = load_predictions(Path(args.predictions))
    records     = load_split(Path(args.split))
    results     = evaluate(predictions, records)
    print_results(results)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  Results saved to {args.output}")


if __name__ == "__main__":
    main()
