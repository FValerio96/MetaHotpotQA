"""
Subgraph Patcher: completes existing subgraphs without rebuilding them from
scratch.

Problem solved:
  The subgraph_builder had a cache-handling bug: when the seed node was
  already in cache, only that node was added to local_nodes, ignoring all
  its neighbors (which were also in cache). The resulting subgraphs often
  contained only the seed node, making it nearly impossible to find the
  answer.

What this script does:
  1. For each existing subgraph JSON, identify QIDs referenced in
     props/extra_props of nodes that are NOT present in the nodes list
     (the "missing neighbors").
  2. Recover them from the cache (if available) or download from Wikidata.
  3. Rewrite the updated JSON file.
  4. For fully missing subgraphs (build failures), rebuild from scratch by
     calling the normal builder.

Usage:
    python 03_patch_subgraphs.py [--subgraphs DIR] [--cache-file PATH]
                                 [--ontologies-dir DIR] [--workers N]
                                 [--dataset PATH]
"""

import argparse
import json
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Set

from wikidata_client import (
    fetch_node_text_info,
    extract_prop_edges,
    normalize_nodes_qid_field,
)

# 02_build_subgraphs.py cannot be imported with a normal `import` statement
# (file name starts with a digit). Load it dynamically via importlib.
import importlib.util as _importlib_util
_builder_spec = _importlib_util.spec_from_file_location(
    "_subgraph_builder", Path(__file__).parent / "02_build_subgraphs.py"
)
_builder = _importlib_util.module_from_spec(_builder_spec)
_builder_spec.loader.exec_module(_builder)
ThreadSafeCache = _builder.ThreadSafeCache
RateLimiter = _builder.RateLimiter
load_all_ontologies = _builder.load_all_ontologies
build_subgraph_for_example_safe = _builder.build_subgraph_for_example_safe


_rate_limiter = RateLimiter(max_requests_per_second=8.0)


# ============================================================================
# PATCHING A SINGLE SUBGRAPH
# ============================================================================

def collect_referenced_qids(nodes_dict: Dict[str, dict]) -> Set[str]:
    """
    Collect all QIDs referenced in props/extra_props of any node that are
    not already a key in the dictionary.
    """
    present = set(nodes_dict.keys())
    referenced: Set[str] = set()
    for node in nodes_dict.values():
        for field in ("props", "extra_props"):
            for targets in node.get(field, {}).values():
                referenced.update(targets)
    return referenced - present


def patch_subgraph_file(
    sg_path: Path,
    cache_snap: Dict[str, dict],
) -> tuple[str, int, int]:
    """
    Patch a single subgraph JSON file:
    - Find missing QIDs referenced in props/extra_props
    - Recover them from cache (priority) or from Wikidata
    - Rewrite the file

    The search for missing QIDs is iterative: after adding direct neighbors,
    we check whether their extra_props (if present in cache) reference
    further missing nodes, and so on until closure.
    Nodes added via API (not in cache) are taken as 0-hop (label +
    description) without further expansion, consistently with
    fetch_node_text_info.

    Return (ex_id, total_missing, n_fetched_from_api).
    """
    with open(sg_path) as f:
        sg = json.load(f)

    nodes_dict: Dict[str, dict] = {n["qid"]: n for n in sg.get("nodes", [])}

    total_missing = 0
    total_api_calls = 0

    # Iterative loop: continue while there are referenced but missing QIDs.
    # Nodes added from cache may have extra_props referencing further QIDs,
    # which get discovered in the next iteration.
    # Nodes added from API (fetch_node_text_info) have no props/extra_props,
    # so they do not produce further iterations.
    while True:
        missing_qids = collect_referenced_qids(nodes_dict)
        if not missing_qids:
            break

        total_missing += len(missing_qids)

        from_cache: Dict[str, dict] = {}
        still_missing: Set[str] = set()
        for qid in missing_qids:
            if qid in cache_snap:
                from_cache[qid] = cache_snap[qid]
            else:
                still_missing.add(qid)

        from_api: Dict[str, dict] = {}
        if still_missing:
            _rate_limiter.wait()
            from_api = fetch_node_text_info(still_missing)
            total_api_calls += len(from_api)

        new_nodes = {**from_cache, **from_api}
        if not new_nodes:
            break  # no recoverable node, exit

        nodes_dict.update(new_nodes)

    if total_missing == 0:
        return sg["id"], 0, 0

    nodes_dict = normalize_nodes_qid_field(nodes_dict)

    # Recompute edges: use props/extra_props of all nodes,
    # keeping only edges whose target is present in nodes_dict.
    prop_edges = extract_prop_edges(nodes_dict)
    type_edges = [
        e for e in sg.get("edges", [])
        if e.get("type") == "TYPE"
    ]

    sg["nodes"] = list(nodes_dict.values())
    sg["edges"] = prop_edges + type_edges

    with open(sg_path, "w", encoding="utf-8") as f:
        json.dump(sg, f, ensure_ascii=False)

    return sg["id"], total_missing, total_api_calls


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def patch_existing(
    subgraphs_dir: Path,
    cache_snap: Dict[str, dict],
    max_workers: int,
) -> dict:
    """Patch all existing subgraph JSON files in parallel."""
    sg_files = list(subgraphs_dir.glob("*.json"))
    total = len(sg_files)
    print(f"[*] Subgraphs to patch: {total}")

    stats = {"patched": 0, "already_ok": 0, "errors": 0, "api_calls": 0}
    start = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(patch_subgraph_file, p, cache_snap): p for p in sg_files}
        for i, future in enumerate(as_completed(futures), 1):
            try:
                ex_id, n_missing, n_api = future.result()
                if n_missing > 0:
                    stats["patched"] += 1
                    stats["api_calls"] += n_api
                else:
                    stats["already_ok"] += 1
            except Exception as e:
                stats["errors"] += 1
                print(f"[!] Error {futures[future].name}: {e}")

            if i % 1000 == 0:
                elapsed = time.time() - start
                print(f"    [{i}/{total}] {i/elapsed:.1f}/s  "
                      f"patched:{stats['patched']} ok:{stats['already_ok']} "
                      f"err:{stats['errors']} api:{stats['api_calls']}")

    return stats


def build_missing(
    missing_examples: List[dict],
    all_ontologies: dict,
    node_cache: ThreadSafeCache,
    subgraphs_dir: Path,
    max_workers: int,
) -> dict:
    """Build the missing subgraphs from scratch (build failures)."""
    total = len(missing_examples)
    if total == 0:
        print("[*] No missing subgraphs.")
        return {"built": 0, "errors": 0}

    print(f"[*] Missing subgraphs to build: {total}")
    stats = {"built": 0, "errors": 0}
    start = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(
                build_subgraph_for_example_safe,
                example,
                all_ontologies,
                node_cache,
                subgraphs_dir,
            ): example["id"]
            for example in missing_examples
        }
        for i, future in enumerate(as_completed(futures), 1):
            ex_id = futures[future]
            try:
                _, success, err = future.result()
                if success:
                    stats["built"] += 1
                else:
                    stats["errors"] += 1
                    print(f"[!] Failed {ex_id}: {err}")
            except Exception as e:
                stats["errors"] += 1
                print(f"[!] Exception {ex_id}: {e}")

            if i % 50 == 0:
                elapsed = time.time() - start
                print(f"    [{i}/{total}] {i/elapsed:.1f}/s  "
                      f"built:{stats['built']} err:{stats['errors']}")

    return stats


def run(
    subgraphs_dir: str,
    dataset_path: str,
    ontologies_dir: str,
    cache_file: str,
    max_workers: int,
):
    subgraphs_path = Path(subgraphs_dir)
    dataset = Path(dataset_path)
    ontologies_path = Path(ontologies_dir)
    cache_path = Path(cache_file)

    # Load cache
    cache_snap: Dict[str, dict] = {}
    node_cache = ThreadSafeCache()
    if cache_path.exists():
        print(f"[*] Loading cache from {cache_path}...")
        with open(cache_path, "rb") as f:
            cache_snap = pickle.load(f)
        node_cache = ThreadSafeCache(dict(cache_snap))
        print(f"    {len(cache_snap)} cached nodes")
    else:
        print("[!] Cache not found; missing neighbors will all be fetched from Wikidata.")

    # Find missing subgraphs (build failures)
    present_ids = {p.stem for p in subgraphs_path.glob("*.json")}
    missing_examples: List[dict] = []
    with open(dataset) as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            if ex["id"] not in present_ids:
                missing_examples.append(ex)

    print(f"\n[1/2] Patching existing subgraphs ({len(present_ids)} files)...")
    patch_stats = patch_existing(subgraphs_path, cache_snap, max_workers)
    print(f"      Patched: {patch_stats['patched']}  "
          f"Already ok: {patch_stats['already_ok']}  "
          f"Errors: {patch_stats['errors']}  "
          f"API calls: {patch_stats['api_calls']}")

    print(f"\n[2/2] Building missing subgraphs ({len(missing_examples)})...")
    all_ontologies = load_all_ontologies(ontologies_path)
    build_stats = build_missing(missing_examples, all_ontologies, node_cache, subgraphs_path, max_workers)
    print(f"      Built: {build_stats['built']}  Errors: {build_stats['errors']}")

    # Update cache on disk
    if missing_examples:
        snap = node_cache.get_snapshot()
        with open(cache_path, "wb") as f:
            pickle.dump(snap, f)
        print(f"[*] Cache updated: {len(snap)} nodes -> {cache_path}")

    print("\n=== DONE ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patch existing subgraphs by adding missing neighbors.")
    repo_root = Path(__file__).resolve().parent.parent.parent

    parser.add_argument("--subgraphs", default=str(repo_root / "data" / "subgraphs"),
                        help="Directory of subgraphs to patch")
    parser.add_argument("--dataset",
                        default=str(repo_root / "data" / "hotpot_matched.jsonl"),
                        help="Path to the JSONL dataset")
    parser.add_argument("--ontologies-dir", default=str(repo_root / "ontologies"),
                        help="Ontologies directory (for missing subgraphs)")
    parser.add_argument("--cache-file", default=str(repo_root / "data" / "node_cache.pkl"),
                        help="Pickle file of the node cache")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel threads (default: 4)")
    args = parser.parse_args()

    run(
        subgraphs_dir=args.subgraphs,
        dataset_path=args.dataset,
        ontologies_dir=args.ontologies_dir,
        cache_file=args.cache_file,
        max_workers=args.workers,
    )
