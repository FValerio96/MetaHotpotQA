"""
For every bridge record in published_splits/, compares:
  - len(entity_qids)       : seed entities we mapped to Wikidata
  - n_hotpot_pages         : unique Wikipedia pages in HotpotQA supporting_facts

A mismatch (n_hotpot_pages > len(entity_qids)) means at least one Wikipedia page
failed to map to a Wikidata QID in our pipeline.

Prerequisites:
  - published_splits/ at the repository root (shipped with the repo).
  - hotpot_train.json and hotpot_dev.json downloaded into _hotpot_cache/.
    The demo (demo/run_pipeline_demo.py) downloads these automatically into
    demo/_persistent_cache/ — symlink or copy them to _hotpot_cache/ before
    running this script.

Output: entity_coverage.tsv next to this script (only mismatches).
Summary printed to stdout.
"""

import json
import os

REPO_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUB_DIR      = os.path.join(REPO_ROOT, "published_splits")
OUT_FILE     = os.path.join(os.path.dirname(__file__), "entity_coverage.tsv")
CACHE_DIR    = os.path.join(os.path.dirname(__file__), "_hotpot_cache")

HOTPOT_FILES = {
    "train": os.path.join(CACHE_DIR, "hotpot_train.json"),
    "dev":   os.path.join(CACHE_DIR, "hotpot_dev.json"),
}
SPLITS = ["train", "dev", "test"]


def load_hotpot_index() -> dict[str, dict]:
    index: dict[str, dict] = {}
    for path in HOTPOT_FILES.values():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found. Download hotpot_train.json and hotpot_dev.json "
                f"(e.g. via demo/run_pipeline_demo.py, which caches them under "
                f"demo/_persistent_cache/) and place them in {CACHE_DIR}."
            )
        with open(path) as f:
            for item in json.load(f):
                index[item["_id"]] = item
    return index


def load_published(split: str) -> list[dict]:
    with open(os.path.join(PUB_DIR, f"{split}.jsonl")) as f:
        return [json.loads(l) for l in f]


def main():
    hotpot = load_hotpot_index()

    total = match = mismatch = not_found = 0
    rows = []

    for split in SPLITS:
        for r in load_published(split):
            if r["reasoning_type"] != "bridge":
                continue
            total += 1
            item = hotpot.get(r["id"])
            if item is None:
                not_found += 1
                continue

            n_hotpot = len({title for title, _ in item.get("supporting_facts", [])})
            n_ours   = len(r["entity_qids"])

            if n_hotpot != n_ours:
                mismatch += 1
                rows.append((split, r["id"], n_ours, n_hotpot, n_hotpot - n_ours,
                             r["answer"], r["question"]))
            else:
                match += 1

    print(f"bridge records checked : {total}")
    print(f"  entity count matches : {match}")
    print(f"  mismatches           : {mismatch}  ({100*mismatch/total:.1f}%)")
    print(f"  not found in hotpot  : {not_found}")

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, "w") as f:
        f.write("split\tid\tn_ours\tn_hotpot\tmissing\tanswer\tquestion\n")
        for row in sorted(rows, key=lambda x: -x[4]):
            f.write("\t".join(str(x) for x in row) + "\n")

    if mismatch:
        print(f"\nDetails in {OUT_FILE}")


if __name__ == "__main__":
    main()
