"""
Stage 1b — Match each HotpotQA question to one or more Text2KGBench ontologies.

For every record produced by Stage 1a, intersects the seed entities' QIDs and
P31 classes with the concept QIDs of each ontology. Records are partitioned
into three outputs:

  - <out>           : at least one ontology matched on a non-Q5 QID (kept).
  - <out>.q5_only   : every match collapsed to {Q5} (human); excluded because
                      Q5 appears across all ontologies and provides no
                      discriminative domain signal (Section 3.1 of the paper).
  - dropped         : no ontology matched (silently filtered, count printed).

Schema of the kept output:
  {
    "id": str,                              # HotpotQA question id
    "question": str,
    "answer": str,
    "entities": [{title, sentence_idx, qid, classes}, ...],
    "matched_ontologies": [
        {"ont_id": "ont_2_music", "matched_qids": ["Q482994", ...]},
        ...
    ],
    ... (all other fields from Stage 1a are preserved)
  }

Usage:
  python 02_match_ontologies.py \\
      --in  data/hotpot_linked.jsonl \\
      --ontologies ontologies/extended \\
      --out data/hotpot_matched.jsonl
"""

import argparse
import json
from pathlib import Path


def load_ontology_concept_qids(ontology_path: Path) -> set[str]:
    with open(ontology_path) as f:
        ontology = json.load(f)
    return {
        c["qid"]
        for c in ontology.get("concepts", [])
        if isinstance(c.get("qid"), str) and c["qid"].startswith("Q")
    }


def get_ontology_id(ontology_path: Path) -> str:
    with open(ontology_path) as f:
        ont_id = json.load(f).get("id")
    if ont_id:
        return ont_id
    parts = ontology_path.stem.split("_")
    return f"ont_{parts[0]}_{parts[1]}" if parts[0].isdigit() else ontology_path.stem


def extract_qids(example: dict) -> set[str]:
    qids: set[str] = set()
    for e in example.get("entities", []):
        if e.get("qid"):
            qids.add(e["qid"])
        qids.update(c for c in e.get("classes", []) if isinstance(c, str))
    return qids


def load_ontologies(ontologies_dir: Path) -> dict[str, set[str]]:
    onts = {}
    for path in sorted(ontologies_dir.glob("*.json")):
        onts[get_ontology_id(path)] = load_ontology_concept_qids(path)
    if not onts:
        raise FileNotFoundError(f"No ontology JSON files in {ontologies_dir}")
    return onts


def match_record(example: dict, ontologies: dict[str, set[str]]) -> tuple[list[dict], set[str]]:
    ex_qids = extract_qids(example)
    matched, all_matched_qids = [], set()
    for ont_id, ont_qids in ontologies.items():
        intersection = ex_qids & ont_qids
        if intersection:
            matched.append({"ont_id": ont_id, "matched_qids": sorted(intersection)})
            all_matched_qids.update(intersection)
    return matched, all_matched_qids


def run(in_path: Path, ontologies_dir: Path, out_path: Path) -> None:
    ontologies = load_ontologies(ontologies_dir)
    print(f"Loaded {len(ontologies)} ontologies from {ontologies_dir}")

    out_q5 = out_path.with_suffix(out_path.suffix + ".q5_only")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = no_match = q5_only = kept = 0
    with open(in_path) as fin, open(out_path, "w") as fout, open(out_q5, "w") as fq5:
        for line_num, line in enumerate(fin, start=1):
            if not line.strip():
                continue
            total += 1
            example = json.loads(line)
            matched, all_qids = match_record(example, ontologies)

            if not matched:
                no_match += 1
                continue

            example["matched_ontologies"] = matched
            if all_qids == {"Q5"}:
                q5_only += 1
                fq5.write(json.dumps(example, ensure_ascii=False) + "\n")
            else:
                kept += 1
                fout.write(json.dumps(example, ensure_ascii=False) + "\n")

            if line_num % 5000 == 0:
                print(f"  processed {line_num}")

    print(f"\nTotal:    {total}")
    print(f"No match: {no_match}  ({100*no_match/total:.1f}%) — dropped")
    print(f"Q5 only:  {q5_only}   ({100*q5_only/total:.1f}%) → {out_q5}")
    print(f"Kept:     {kept}      ({100*kept/total:.1f}%) → {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", required=True, type=Path,
                   help="Input JSONL from Stage 1a (hotpot_linked.jsonl).")
    p.add_argument("--ontologies", required=True, type=Path,
                   help="Directory containing the 10 extended ontology JSON files.")
    p.add_argument("--out", required=True, type=Path,
                   help="Output JSONL path; a sibling .q5_only file is also written.")
    args = p.parse_args()
    run(args.in_path, args.ontologies, args.out)


if __name__ == "__main__":
    main()
