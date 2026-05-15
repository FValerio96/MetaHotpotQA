"""
Creates published_splits/ from splits_final/ by removing internal-only fields.

Removed fields:
  - context        : original HotpotQA Wikipedia text (reobtainable from HotpotQA)
  - entities       : seed entity details (reobtainable by joining with HotpotQA on id)
  - answer_aliases : always identical to [answer], was never extended
  - path_found     : redundant with supporting_paths being non-empty
  - split          : redundant with the file name (train/dev/test)

supporting_paths is intentionally kept: required by llm_baseline.py to build
oracle KG context and by evaluate_oracle_breakdown.py to classify oracle_full
vs oracle_partial.
"""

import argparse
import json
import os

DEFAULT_SRC = os.path.join(os.path.dirname(__file__), "splits_final")
DEFAULT_DST = os.path.join(os.path.dirname(__file__), "published_splits")

FIELDS_TO_REMOVE = {"context", "entities", "answer_aliases", "path_found", "split"}

SPLITS = ["train", "dev", "test"]


def publish_split(split: str, src_dir: str, dst_dir: str) -> int:
    src = os.path.join(src_dir, f"{split}.jsonl")
    dst = os.path.join(dst_dir, f"{split}.jsonl")

    count = 0
    with open(src) as f_in, open(dst, "w") as f_out:
        for line in f_in:
            record = json.loads(line)
            clean = {k: v for k, v in record.items() if k not in FIELDS_TO_REMOVE}
            clean["matched_ontologies"] = [
                ont for ont in clean.get("matched_ontologies", [])
                if ont.get("matched_qids") != ["Q5"]
            ]
            f_out.write(json.dumps(clean, ensure_ascii=False) + "\n")
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Strip internal-only fields from splits_final/ to produce published_splits/"
    )
    parser.add_argument(
        "--in", dest="src_dir", default=DEFAULT_SRC,
        help=f"Input splits directory (default: {DEFAULT_SRC})",
    )
    parser.add_argument(
        "--out", dest="dst_dir", default=DEFAULT_DST,
        help=f"Output published-splits directory (default: {DEFAULT_DST})",
    )
    args = parser.parse_args()

    os.makedirs(args.dst_dir, exist_ok=True)
    for split in SPLITS:
        n = publish_split(split, args.src_dir, args.dst_dir)
        print(f"{split}: {n} records -> {args.dst_dir}/{split}.jsonl")


if __name__ == "__main__":
    main()
