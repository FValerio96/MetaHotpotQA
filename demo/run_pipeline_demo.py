"""
End-to-end pipeline demo for MetaHotpotQA.

Runs all 12 pipeline modules on a tiny sample of records (default 3) to
demonstrate the construction pipeline and to let reviewers verify
reproducibility without running the full multi-day pipeline.

Modes:
  --mode fast   (default)  Skips the LLM-based Tier 3 in answer coverage
                           by only sampling records with match_tier == 1.
                           Requires only Wikidata API access. ~5–10 min.
  --mode full              Samples records irrespective of match_tier and
                           uses Ollama (qwen3:14b-q4_K_M) for Tier 3.
                           Requires Ollama running locally. ~10–20 min.

The sample always includes at least one record with strategy_used == 1,
so the bridge patcher (Stage 4b) is exercised on a non-trivial input.

Workspace: demo/_workspace/  (cleared at start of each run)

Verification: at the end, the regenerated published_splits/ for the sampled
IDs is diffed against the original published_splits/. Mismatches on the
canonical fields (entity_qids, answer_node_qid, reasoning_type, difficulty,
strategy_used, match_tier) are reported.

Usage:
  python demo/run_pipeline_demo.py
  python demo/run_pipeline_demo.py --mode full --n 5
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import urllib.request
import urllib.error
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PIPELINE  = REPO_ROOT / "pipeline"
WORKSPACE = REPO_ROOT / "demo" / "_workspace"
GROUND_TRUTH_SPLITS = REPO_ROOT / "published_splits"
ONTOLOGIES_SRC = REPO_ROOT / "ontologies"
PERSISTENT_CACHE = REPO_ROOT / "demo" / "_persistent_cache"  # node_cache.pkl + hotpot full

CANONICAL_FIELDS = (
    "id", "question", "answer", "answer_node_qid", "entity_qids",
    "reasoning_type", "difficulty", "strategy_used", "match_tier",
)

HOTPOT_URLS = {
    "train": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json",
    "dev":   "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
}


# ──────────────────────────────────────────────────────────────────────────
# Pre-flight
# ──────────────────────────────────────────────────────────────────────────

def check_wikidata(contact: str) -> None:
    print("  [pre] Wikidata API ... ", end="", flush=True)
    ua = f"MetaHotpotQA-demo/1.0 (https://github.com/FValerio96/MetaHotpot; mailto:{contact})"
    try:
        req = urllib.request.Request(
            "https://www.wikidata.org/w/api.php?action=wbgetentities&ids=Q42&format=json",
            headers={"User-Agent": ua},
        )
        urllib.request.urlopen(req, timeout=10).read()
        print("OK")
    except (urllib.error.URLError, TimeoutError) as e:
        print(f"FAIL ({e})")
        sys.exit(1)


def check_ollama(model: str) -> None:
    print(f"  [pre] Ollama with model '{model}' ... ", end="", flush=True)
    try:
        body = urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=5).read()
        models = [m["name"] for m in json.loads(body).get("models", [])]
        if not any(model in m for m in models):
            print(f"FAIL (model not found; available: {models})")
            sys.exit(1)
        print("OK")
    except (urllib.error.URLError, TimeoutError) as e:
        print(f"FAIL ({e})")
        sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────
# Sampling
# ──────────────────────────────────────────────────────────────────────────

def load_published_test() -> list[dict]:
    path = GROUND_TRUTH_SPLITS / "test.jsonl"
    if not path.exists():
        print(f"FAIL: ground truth missing at {path}")
        sys.exit(1)
    with open(path) as f:
        return [json.loads(l) for l in f]


def sample_records(records: list[dict], n: int, fast: bool, seed: int) -> list[dict]:
    rng = random.Random(seed)
    pool = [r for r in records if r["match_tier"] == 1] if fast else records

    waypoint = [r for r in pool if r.get("strategy_used") == 1]
    if not waypoint:
        print("FAIL: no record with strategy_used==1 found in pool (cannot exercise bridge patcher)")
        sys.exit(1)

    forced = rng.choice(waypoint)
    others = [r for r in pool if r["id"] != forced["id"]]
    rest = rng.sample(others, n - 1) if n > 1 else []
    chosen = [forced] + rest
    print(f"  Sampled {len(chosen)} records (incl. 1 with strategy_used==1):")
    for r in chosen:
        print(f"    {r['id']}  {r['reasoning_type']:11} {r['difficulty']:20} tier={r['match_tier']}")
    return chosen


# ──────────────────────────────────────────────────────────────────────────
# Workspace setup
# ──────────────────────────────────────────────────────────────────────────

def reset_workspace() -> None:
    if WORKSPACE.exists():
        shutil.rmtree(WORKSPACE)
    (WORKSPACE / "data").mkdir(parents=True)
    (WORKSPACE / "ontologies").mkdir()
    for f in ONTOLOGIES_SRC.iterdir():
        shutil.copy(f, WORKSPACE / "ontologies" / f.name)
    PERSISTENT_CACHE.mkdir(parents=True, exist_ok=True)
    # Seed workspace's node_cache from persistent location, if present.
    persistent_node_cache = PERSISTENT_CACHE / "node_cache.pkl"
    if persistent_node_cache.exists():
        shutil.copy(persistent_node_cache, WORKSPACE / "data" / "node_cache.pkl")
        size_kb = persistent_node_cache.stat().st_size // 1024
        print(f"  Seeded node_cache.pkl from persistent cache ({size_kb} KB)")


def save_node_cache_persistent() -> None:
    """Copy the workspace node_cache back to persistent location for future runs."""
    src = WORKSPACE / "data" / "node_cache.pkl"
    if src.exists():
        shutil.copy(src, PERSISTENT_CACHE / "node_cache.pkl")
        size_kb = src.stat().st_size // 1024
        print(f"  Persisted node_cache.pkl ({size_kb} KB)")


def prepare_hotpot_subset(target_ids: set[str]) -> None:
    """Download HotpotQA once into a persistent cache, then subset to target_ids."""
    workspace_cache = WORKSPACE / "data" / "_hotpot_cache"
    workspace_cache.mkdir(parents=True)

    found_ids: set[str] = set()
    for split, url in HOTPOT_URLS.items():
        full = PERSISTENT_CACHE / f"hotpot_{split}.json"
        if not full.exists():
            print(f"  Downloading {url} (persistent, one-time)...")
            urllib.request.urlretrieve(url, full)
        with open(full) as f:
            all_records = json.load(f)
        subset = [r for r in all_records if r["_id"] in target_ids]
        found_ids.update(r["_id"] for r in subset)
        with open(workspace_cache / f"hotpot_{split}.json", "w") as f:
            json.dump(subset, f)
        print(f"  Subset {split}: {len(subset)} records")

    missing = target_ids - found_ids
    if missing:
        print(f"FAIL: {len(missing)} sampled IDs not found in HotpotQA: {missing}")
        sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────
# Stage runner
# ──────────────────────────────────────────────────────────────────────────

_SUBPROCESS_ENV: dict[str, str] = {}


def run_stage(label: str, cmd: list[str], cwd: Path = WORKSPACE) -> None:
    print(f"\n[{label}] {' '.join(str(c) for c in cmd)}")
    env = {**os.environ, **_SUBPROCESS_ENV}
    r = subprocess.run(cmd, cwd=cwd, env=env)
    if r.returncode != 0:
        print(f"FAIL: stage '{label}' exited with code {r.returncode}")
        sys.exit(1)


def assert_jsonl_count(path: Path, expected: int, label: str) -> None:
    if not path.exists():
        print(f"FAIL [{label}]: missing output {path}")
        sys.exit(1)
    n = sum(1 for _ in open(path))
    print(f"  [check] {label}: {path.relative_to(WORKSPACE)} → {n} records (expected {expected})")
    if n != expected:
        print(f"FAIL [{label}]: count {n} != expected {expected}")
        sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────
# Final verification
# ──────────────────────────────────────────────────────────────────────────

def diff_against_truth(sampled: list[dict]) -> bool:
    out_path = WORKSPACE / "data" / "published_splits"
    regen: dict[str, dict] = {}
    for split in ("train", "dev", "test"):
        p = out_path / f"{split}.jsonl"
        if not p.exists():
            continue
        for line in open(p):
            r = json.loads(line)
            regen[r["id"]] = r

    print(f"\n[diff] Regenerated records: {len(regen)}")
    all_ok = True
    for truth in sampled:
        qid = truth["id"]
        new = regen.get(qid)
        if new is None:
            print(f"  [{qid}] MISSING in regenerated output")
            all_ok = False
            continue
        diffs = []
        for f in CANONICAL_FIELDS:
            if truth.get(f) != new.get(f):
                diffs.append(f"{f}: truth={truth.get(f)!r} regen={new.get(f)!r}")
        if diffs:
            print(f"  [{qid}] MISMATCH:")
            for d in diffs:
                print(f"      {d}")
            all_ok = False
        else:
            print(f"  [{qid}] OK")
    return all_ok


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="End-to-end MetaHotpotQA pipeline demo.",
        epilog="A contact email is required by Wikidata's User-Agent policy: "
               "https://meta.wikimedia.org/wiki/User-Agent_policy",
    )
    p.add_argument("--contact-email", required=True,
                   help="Your email, embedded in the Wikidata User-Agent header.")
    p.add_argument("--mode", choices=["fast", "full"], default="fast")
    p.add_argument("--n", type=int, default=3, help="Sample size (default: 3)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    fast = args.mode == "fast"

    _SUBPROCESS_ENV["WIKIDATA_CONTACT"] = args.contact_email

    print(f"=== MetaHotpotQA pipeline demo (mode={args.mode}, n={args.n}) ===\n")
    print(f"Contact (User-Agent): {args.contact_email}")
    print("Pre-flight:")
    check_wikidata(args.contact_email)
    if not fast:
        check_ollama("qwen3:14b")

    print("\nSampling ground-truth records:")
    truth = load_published_test()
    sampled = sample_records(truth, args.n, fast=fast, seed=args.seed)
    target_ids = {r["id"] for r in sampled}

    print("\nSetting up workspace:")
    reset_workspace()
    prepare_hotpot_subset(target_ids)

    py = sys.executable
    expected_n = len(sampled)

    # Stage 1a — link to Wikidata
    run_stage("1a", [py, str(PIPELINE / "1_ontology_matching" / "01_link_hotpot_to_wikidata.py"),
                     "--cache-dir", "data/_hotpot_cache",
                     "--out", "data/hotpot_linked.jsonl",
                     "--workers", "10"])
    assert_jsonl_count(WORKSPACE / "data" / "hotpot_linked.jsonl", expected_n, "1a")

    # Stage 1b — ontology matching
    run_stage("1b", [py, str(PIPELINE / "1_ontology_matching" / "02_match_ontologies.py"),
                     "--in", "data/hotpot_linked.jsonl",
                     "--ontologies", "ontologies",
                     "--out", "data/hotpot_matched.jsonl"])
    assert_jsonl_count(WORKSPACE / "data" / "hotpot_matched.jsonl", expected_n, "1b")

    # Stage 2 — subgraph extraction (single-threaded for demo to avoid burst rate-limits)
    run_stage("2-build", [py, str(PIPELINE / "2_subgraph_extraction" / "02_build_subgraphs.py"),
                          "--input", "data/hotpot_matched.jsonl",
                          "--ontologies-dir", "ontologies",
                          "--output-dir", "data/subgraphs",
                          "--cache-file", "data/node_cache.pkl",
                          "--workers", "1"])
    n_subgraphs = len(list((WORKSPACE / "data" / "subgraphs").glob("*.json")))
    print(f"  [check] 2-build: {n_subgraphs} subgraph JSONs (expected {expected_n})")
    save_node_cache_persistent()  # always persist progress, even on partial success
    if n_subgraphs != expected_n:
        sys.exit(1)

    # Stage 2 — patch (no-op on freshly built; included for completeness)
    run_stage("2-patch", [py, str(PIPELINE / "2_subgraph_extraction" / "03_patch_subgraphs.py"),
                          "--subgraphs", "data/subgraphs",
                          "--dataset", "data/hotpot_matched.jsonl",
                          "--ontologies-dir", "ontologies",
                          "--cache-file", "data/node_cache.pkl",
                          "--workers", "1"])
    save_node_cache_persistent()

    # Stage 3 — answer coverage (Tier 3 LLM not invoked in fast mode)
    cmd3 = [py, str(PIPELINE / "3_answer_coverage" / "01_find_answers.py"),
            "--dataset", "data/hotpot_matched.jsonl",
            "--subgraphs", "data/subgraphs",
            "--output", "data/answer_search"]
    if not fast:
        cmd3 += ["--ollama-model", "qwen3:14b"]
    run_stage("3", cmd3)
    found_path = WORKSPACE / "data" / "answer_search" / "found.jsonl"
    not_found_path = WORKSPACE / "data" / "answer_search" / "not_found.jsonl"
    n_found = sum(1 for _ in open(found_path)) if found_path.exists() else 0
    n_not_found = sum(1 for _ in open(not_found_path)) if not_found_path.exists() else 0
    print(f"  [check] 3: found={n_found} not_found={n_not_found} (expected {expected_n} found)")
    if n_found != expected_n:
        sys.exit(1)

    # Stage 4 — annotate paths
    run_stage("4a", [py, str(PIPELINE / "4_path_annotation" / "01_annotate_paths.py"),
                     "--found", "data/answer_search/found.jsonl",
                     "--subgraphs", "data/subgraphs",
                     "--output", "data/path_annotations"])
    assert_jsonl_count(WORKSPACE / "data" / "path_annotations" / "path_annotations.jsonl",
                       expected_n, "4a")

    # Stage 5a (initial) — build splits, needed as input for bridge patcher
    run_stage("5a-pre", [py, str(PIPELINE / "5_splits" / "01_build_splits.py"),
                         "--annotations", "data/path_annotations/path_annotations.jsonl",
                         "--found", "data/answer_search/found.jsonl",
                         "--output", "data/splits_internal"])

    # Stage 4b — bridge patcher (operates on splits + subgraphs)
    run_stage("4b", [py, str(PIPELINE / "4_path_annotation" / "02_bridge_patcher.py"),
                     "--splits", "data/splits_internal",
                     "--subgraphs", "data/subgraphs",
                     "--output-splits", "data/splits_patched",
                     "--output-subgraphs", "data/subgraphs_patched"])

    # Stage 4c — filter no_path (in-place on patched splits)
    run_stage("4c", [py, str(PIPELINE / "4_path_annotation" / "03_filter_no_path.py"),
                     "--splits", "data/splits_patched"])

    # Stage 2 — merge final
    run_stage("2-merge", [py, str(PIPELINE / "2_subgraph_extraction" / "04_merge_final.py"),
                          "--extended", "data/subgraphs",
                          "--patched", "data/subgraphs_patched",
                          "--output", "data/subgraphs_final",
                          "--force"])

    # Stage 5b — publish
    run_stage("5b", [py, str(PIPELINE / "5_splits" / "02_publish_splits.py"),
                     "--in", "data/splits_patched",
                     "--out", "data/published_splits"])

    print("\n=== Verification ===")
    ok = diff_against_truth(sampled)
    print()
    if ok:
        print("ALL OK — regenerated records match the published ground truth byte-for-byte on canonical fields.")
        sys.exit(0)
    else:
        print("MISMATCH — see report above.")
        sys.exit(2)


if __name__ == "__main__":
    main()
