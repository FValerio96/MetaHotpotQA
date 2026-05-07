"""
01_build_splits.py — Build train/dev/test splits for MetaHotpotQA.

Joins path_annotations.jsonl (paths and difficulty) with found.jsonl
(context and full entities) into unified records, then performs a 70/15/15
stratified split preserving the difficulty distribution across all three
splits.

Output record structure:
    {
      "id":               str,    # HotpotQA question ID
      "question":         str,    # question text
      "answer":           str,    # textual answer (GT for text-only systems)
      "answer_node_qid":  str,    # Wikidata QID of the answer node (GT for KG-based)
      "answer_aliases":   [...],  # contains at least the HotpotQA textual GT;
                                  # extensible with Wikidata labels/aliases for
                                  # robust string-match evaluation
      "entities":         [...],  # full entity list with title/qid/sentence_idx
      "entity_qids":      [...],  # deduplicated QIDs (seed nodes, from path_annotations)
      "context":          [...],  # HotpotQA supporting paragraphs
      "reasoning_type":   str,    # "bridge" | "comparison"
      "difficulty":       str,    # "entity_selection" | "traversal" |
                                  # "property_comparison"
                                  # ("no_path" records are filtered out before
                                  # the split — see --keep-no-path)
      "strategy_used":    int,    # 0=entity_sel, 1=waypoint, 2=direct
      "path_found":       bool,
      "match_tier":       int,    # 1=exact, 2=fuzzy, 3=LLM-verified
      "matched_ontologies": [...],
      "supporting_paths": [...],  # bridge: list of paths (empty if S0 or S3)
      "comparison_triples": {...},# comparison: 1-hop properties per entity
      "split":            str,    # "train" | "dev" | "test"
    }

Evaluation notes:
    - KG-based systems: use answer_node_qid as GT, evaluate with Hits@1 on QID.
      This avoids mismatches between Wikidata labels and HotpotQA text
      (e.g. "Robert Downey Jr." vs "Robert Downey Junior").
    - Text-only systems: use answer as GT, evaluate with EM and token-level F1.
    - Path-supervised systems: use supporting_paths as additional supervision
      for records with strategy_used in {1, 2}. S0 (entity_selection) records
      have no path and contribute only as entity-selection examples.
    - answer_aliases: always contains at least the HotpotQA answer string as
      base GT. Extendable via the Wikidata API to add labels and aliases of
      the answer node (e.g. "Robert Downey Jr." besides "Robert Downey Junior").
      Evaluation code should always check against answer_aliases.

Usage:
    python 01_build_splits.py [--annotations PATH] [--found PATH] [--output DIR]
                              [--train-ratio F] [--dev-ratio F] [--seed N]
"""

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[build_splits] %(message)s")
log = logging.getLogger(__name__)


def load_found(path: Path) -> dict[str, dict]:
    """Load found.jsonl deduplicating by ID (keeping the first occurrence)."""
    records: dict[str, dict] = {}
    duplicates = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["id"] in records:
                duplicates += 1
            else:
                records[r["id"]] = r
    log.info(f"found.jsonl: {len(records)} unique records ({duplicates} duplicates dropped)")
    return records


def load_annotations(path: Path) -> dict[str, dict]:
    """Load path_annotations.jsonl indexed by ID."""
    records: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            records[r["id"]] = r
    log.info(f"path_annotations.jsonl: {len(records)} records")
    return records


def merge_record(found: dict, ann: dict) -> dict:
    """
    Merge fields from found.jsonl and path_annotations.jsonl into one record.
    path_annotations is the authoritative source for difficulty, strategy_used,
    path_found, entity_qids, and paths. found.jsonl contributes the full
    context and entities.
    """
    record = {
        "id":                 ann["id"],
        "question":           ann["question"],
        "answer":             ann["answer"],
        "answer_node_qid":    ann["answer_node_qid"],
        "answer_aliases":     [found["answer"]],  # HotpotQA GT as base; extensible with Wikidata labels/aliases
        "entities":           found.get("entities", []),
        "entity_qids":        ann.get("entity_qids", []),
        "context":            found.get("context", []),
        "reasoning_type":     ann["reasoning_type"],
        "difficulty":         ann["difficulty"],
        "strategy_used":      ann["strategy_used"],
        "path_found":         ann["path_found"],
        "match_tier":         ann.get("match_tier"),
        "matched_ontologies": found.get("matched_ontologies", []),
    }

    # Question-type-specific fields
    if ann["reasoning_type"] == "comparison":
        record["comparison_triples"] = ann.get("comparison_triples", {})
        record["supporting_paths"] = []
    else:
        record["supporting_paths"] = ann.get("supporting_paths", [])

    return record


def stratified_split(
    records: list[dict],
    train_ratio: float,
    dev_ratio: float,
    seed: int,
) -> tuple[list, list, list]:
    """
    Stratified split by difficulty x reasoning_type.
    Guarantees that each split preserves the original proportions of every
    stratum.
    """
    rng = random.Random(seed)
    test_ratio = 1.0 - train_ratio - dev_ratio

    # Group by stratum
    strata: dict[str, list] = defaultdict(list)
    for r in records:
        key = f"{r['difficulty']}__{r['reasoning_type']}"
        strata[key].append(r)

    train, dev, test = [], [], []
    for key, group in sorted(strata.items()):
        rng.shuffle(group)
        n = len(group)
        n_train = max(1, round(n * train_ratio))
        n_dev   = max(1, round(n * dev_ratio))
        # test takes the remainder to ensure every record is assigned
        n_test  = n - n_train - n_dev
        if n_test < 0:
            # stratum too small: assign at least 1 to train
            n_train = n
            n_dev, n_test = 0, 0

        train.extend(group[:n_train])
        dev.extend(group[n_train:n_train + n_dev])
        test.extend(group[n_train + n_dev:])

        log.info(
            f"  {key}: {n} total -> train={n_train}, dev={n_dev}, test={n - n_train - n_dev}"
        )

    return train, dev, test


def write_split(records: list[dict], path: Path, split_name: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            r["split"] = split_name
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info(f"  {split_name}: {len(records)} records -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build stratified train/dev/test splits for MetaHotpotQA"
    )
    parser.add_argument(
        "--annotations",
        default="path_annotations_final/path_annotations.jsonl",
        help="Path to path_annotations.jsonl. Default: %(default)s",
    )
    parser.add_argument(
        "--found",
        default="../subgraph_step/answer_search_output_for_extended_version/found.jsonl",
        help="Path to found.jsonl (extended). Default: %(default)s",
    )
    parser.add_argument(
        "--output",
        default="splits",
        help="Output directory. Default: %(default)s",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.70,
        help="Training-set ratio (default: 0.70)",
    )
    parser.add_argument(
        "--dev-ratio", type=float, default=0.15,
        help="Dev-set ratio (default: 0.15)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Reproducibility seed (default: 42)",
    )
    parser.add_argument(
        "--keep-no-path", action="store_true",
        help="Keep records with difficulty='no_path' in the splits. By "
             "default they are filtered out: no KG-grounded system can "
             "navigate to the answer for these records (oracle context would "
             "be empty), so they are only included for diagnostic analysis.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load sources ---
    found = load_found(Path(args.found))
    annotations = load_annotations(Path(args.annotations))

    # --- Merge ---
    merged = []
    missing_found = 0
    for qid, ann in annotations.items():
        if qid not in found:
            missing_found += 1
            continue
        merged.append(merge_record(found[qid], ann))

    if missing_found:
        log.warning(f"{missing_found} annotation records without matching found.jsonl entry")
    log.info(f"Merged records: {len(merged)}")

    # --- Filter no_path records (default on) ---
    if not args.keep_no_path:
        before = len(merged)
        merged = [r for r in merged if r.get("difficulty") != "no_path"]
        removed = before - len(merged)
        log.info(f"'no_path' records removed: {removed} "
                 f"(to keep them, re-run with --keep-no-path)")

    # --- Pre-split statistics ---
    diff_counts: dict[str, int] = defaultdict(int)
    for r in merged:
        diff_counts[r["difficulty"]] += 1
    log.info("Difficulty distribution:")
    for k, v in sorted(diff_counts.items()):
        log.info(f"  {k}: {v} ({100*v/len(merged):.1f}%)")

    # --- Stratified split ---
    log.info(f"Stratified split (seed={args.seed}): {args.train_ratio:.0%} / {args.dev_ratio:.0%} / {1-args.train_ratio-args.dev_ratio:.0%}")
    train, dev, test = stratified_split(merged, args.train_ratio, args.dev_ratio, args.seed)

    # --- Write ---
    log.info("Writing splits:")
    write_split(train, output_dir / "train.jsonl", "train")
    write_split(dev,   output_dir / "dev.jsonl",   "dev")
    write_split(test,  output_dir / "test.jsonl",  "test")

    # --- Summary ---
    summary = {
        "total": len(merged),
        "train": len(train),
        "dev":   len(dev),
        "test":  len(test),
        "seed":  args.seed,
        "ratios": {
            "train": args.train_ratio,
            "dev":   args.dev_ratio,
            "test":  round(1 - args.train_ratio - args.dev_ratio, 2),
        },
        "difficulty_distribution": dict(diff_counts),
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log.info(f"Summary -> {summary_path}")


if __name__ == "__main__":
    main()
