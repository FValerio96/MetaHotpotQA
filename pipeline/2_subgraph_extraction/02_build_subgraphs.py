"""
Subgraph Builder: generates Wikidata subgraphs for each MetaHotpotQA question.

For each question:
1. Load the ontologies it matched with
2. For each QID of the question:
   - If it matches at least one ontology -> follow the relations of ALL matched ontologies
   - Otherwise -> classic 1-hop
3. Save the subgraph as JSON

Features:
- Multi-threading via ThreadPoolExecutor
- Rate limiting and retry with backoff on 429 errors
- Thread-safe cache with lock
- Automatic resume
"""

import json
import os
import pickle
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Set

from wikidata_client import (
    wikidata_get_entities,
    extract_targets,
    fetch_types,
    fetch_node_text_info,
    download_node_1hop,
    download_node_with_ontology_props,
    extend_neighbors_with_concepts,
    merge_node_dicts,
    merge_into_cache,
    extract_prop_edges,
    edges_to_dict_list,
    normalize_nodes_qid_field,
)


# ============================================================================
# RATE LIMITING AND RETRY
# ============================================================================

class RateLimiter:
    """
    Thread-safe rate limiter to throttle API requests.
    """
    def __init__(self, max_requests_per_second: float = 10.0):
        self.min_interval = 1.0 / max_requests_per_second
        self.last_request_time = 0.0
        self.lock = threading.Lock()

    def wait(self):
        """Wait, if necessary, to respect the rate limit."""
        with self.lock:
            now = time.time()
            elapsed = now - self.last_request_time
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self.last_request_time = time.time()


class ThreadSafeCache:
    """
    Thread-safe cache for Wikidata nodes.
    """
    def __init__(self, initial_data: Dict[str, dict] = None):
        self.cache: Dict[str, dict] = initial_data or {}
        self.lock = threading.Lock()
    
    def get(self, qid: str) -> dict:
        with self.lock:
            return self.cache.get(qid)
    
    def contains(self, qid: str) -> bool:
        with self.lock:
            return qid in self.cache
    
    def update(self, nodes: Dict[str, dict]):
        with self.lock:
            for qid, node in nodes.items():
                if qid in self.cache:
                    self.cache[qid] = merge_node_dicts(self.cache[qid], node)
                else:
                    self.cache[qid] = node
    
    def get_snapshot(self) -> Dict[str, dict]:
        with self.lock:
            return dict(self.cache)
    
    def __len__(self):
        with self.lock:
            return len(self.cache)


# Global rate limiter shared across threads
_rate_limiter = RateLimiter(max_requests_per_second=8.0)


def api_call_with_retry(func, *args, max_retries: int = 5, **kwargs):
    """
    Run an API call with retry and exponential backoff.
    Handles 429 (Too Many Requests) errors and timeouts.
    """
    import requests
    
    for attempt in range(max_retries):
        _rate_limiter.wait()
        
        try:
            return func(*args, **kwargs)
        
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                # Too Many Requests - exponential backoff
                wait_time = (2 ** attempt) + (time.time() % 1)  # 1, 2, 4, 8, 16 + jitter
                print(f"    [429] Rate limited, waiting {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                raise
        
        except requests.exceptions.Timeout:
            # Increasing backoff: 5, 10, 20, 40, 80s + jitter
            wait_time = (5 * (2 ** attempt)) + (time.time() % 2)
            print(f"    [Timeout] Waiting {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(wait_time)

        except requests.exceptions.ConnectionError:
            # Increasing backoff: 10, 20, 40, 80, 160s + jitter
            wait_time = (10 * (2 ** attempt)) + (time.time() % 2)
            print(f"    [Connection Error] Waiting {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(wait_time)
    
    raise Exception(f"Max retries ({max_retries}) exceeded for API call")


# ============================================================================
# ONTOLOGY HELPERS
# ============================================================================


def normalize_range(range_val) -> List[str]:
    """
    Normalize the range field, which may be:
    - empty string "" -> []
    - single string "Q5" -> ["Q5"]
    - list ["Q5", "Q618779"] -> ["Q5", "Q618779"]
    """
    if not range_val:
        return []
    if isinstance(range_val, str):
        return [range_val] if range_val else []
    return list(range_val)


def load_all_ontologies(ontologies_dir: Path) -> Dict[str, dict]:
    """
    Load all ontologies from the directory.
    Returns: {ont_id: {relations, concepts, concepts_cfg}}

    Note: normalizes the 'range' field of relations into a list.
    """
    ontologies = {}
    for onto_file in sorted(ontologies_dir.glob("*_ontology.json")):
        with open(onto_file, "r", encoding="utf-8") as f:
            ont = json.load(f)
        
        ont_id = ont.get("id", onto_file.stem)
        
        # Normalize relations (range as a list)
        relations = []
        for rel in ont.get("relations", []):
            relations.append({
                "pid": rel["pid"],
                "label": rel.get("label", ""),
                "domain": rel["domain"],
                "range": normalize_range(rel.get("range", [])),
            })
        
        ontologies[ont_id] = {
            "relations": relations,
            "concepts": {c["qid"] for c in ont.get("concepts", [])},
            # concepts_to_extend is optional (present only in some ontologies)
            "concepts_cfg": {
                c["qid"]: c["follow_pids"] 
                for c in ont.get("concepts_to_extend", [])
            },
        }
    return ontologies


def merge_ontology_configs(ont_ids: List[str], all_ontologies: Dict[str, dict]) -> dict:
    """
    Merge the configurations of multiple ontologies into one.
    """
    merged_relations: List[Dict] = []
    merged_concepts: Set[str] = set()
    merged_concepts_cfg: Dict[str, List[str]] = {}

    seen_rels = set()  # to avoid duplicates (pid, domain)

    for ont_id in ont_ids:
        if ont_id not in all_ontologies:
            continue
        ont = all_ontologies[ont_id]

        # Merge relations
        for rel in ont["relations"]:
            key = (rel["pid"], rel["domain"])
            if key not in seen_rels:
                merged_relations.append(rel)
                seen_rels.add(key)

        # Merge concepts
        merged_concepts.update(ont["concepts"])

        # Merge concepts_cfg (merge follow_pids for the same qid)
        for qid, pids in ont["concepts_cfg"].items():
            if qid in merged_concepts_cfg:
                merged_concepts_cfg[qid] = list(set(merged_concepts_cfg[qid] + pids))
            else:
                merged_concepts_cfg[qid] = pids
    
    return {
        "relations": merged_relations,
        "concepts": merged_concepts,
        "concepts_cfg": merged_concepts_cfg,
    }


def download_node_multi_ontology(
    qid: str,
    classes: List[str],
    merged_config: dict,
) -> Dict[str, dict]:
    """
    Download a node using the merged configuration of multiple ontologies.
    If the QID is in the domain of at least one relation -> use ontology-driven.
    Otherwise -> classic 1-hop.
    """
    relations = merged_config["relations"]
    concepts_cfg = merged_config["concepts_cfg"]

    qid_classes = set(classes)
    domains = {rel["domain"] for rel in relations}

    if qid_classes & domains:
        # Ontology-driven: only follows relations of the ontologies
        node = download_node_with_ontology_props(qid, relations)
        neighbor_qids = {t for targets in node["props"].values() for t in targets}
        neighbor_nodes = fetch_node_text_info(neighbor_qids)

        # Extend neighbors with concepts_to_extend (if present)
        extended_neighbors = {}
        if concepts_cfg:
            extended_neighbors = extend_neighbors_with_concepts(node, concepts_cfg)
            extra_qids = {
                t for ext in extended_neighbors.values()
                for targets in ext["extra_props"].values()
                for t in targets
            }
            extra_qids -= ({qid} | neighbor_qids)
            extra_nodes = fetch_node_text_info(extra_qids)
        else:
            extra_nodes = {}
        
        nodes = {qid: node, **neighbor_nodes, **extra_nodes}
        for q, ext in extended_neighbors.items():
            nodes[q] = merge_node_dicts(nodes.get(q, {}), ext)
        
        return normalize_nodes_qid_field(nodes)
    else:
        # Classic 1-hop
        return download_node_1hop(qid)


def add_type_edges(qid: str, classes: List[str], ontology_concepts: Set[str]) -> List[tuple]:
    """
    Create TYPE edges towards the ontology's concepts.
    """
    return [(qid, "TYPE", c) for c in classes if c in ontology_concepts]


def build_subgraph_for_example_safe(
    example: dict,
    all_ontologies: Dict[str, dict],
    node_cache: ThreadSafeCache,
    output_path: Path,
) -> tuple:
    """
    Thread-safe version of build_subgraph_for_example.
    Returns (ex_id, success, error_msg).
    """
    ex_id = example["id"]

    try:
        # Extract matched ontologies for this question
        matched_ont_ids = [m["ont_id"] for m in example.get("matched_ontologies", [])]

        # Merge configurations of the matched ontologies
        merged_config = merge_ontology_configs(matched_ont_ids, all_ontologies)
        ontology_concepts = merged_config["concepts"]

        local_nodes: Dict[str, dict] = {}
        type_edges: List[tuple] = []

        # Process every entity of the question
        for entity in example.get("entities", []):
            qid = entity.get("qid")
            if not qid or not qid.startswith("Q"):
                continue

            classes = entity.get("classes", [])

            # Thread-safe cache lookup
            cached_node = node_cache.get(qid)
            if cached_node:
                local_nodes[qid] = cached_node
                # Also pull neighbors and their extras from the cache.
                # When a node is downloaded for the first time, all nodes
                # (seed + neighbors + extras) are saved to cache. If we
                # only take the seed node, the subgraph ends up nearly
                # empty (no neighbors -> no edges -> answer not found).
                cache_snap = node_cache.get_snapshot()
                neighbor_qids: Set[str] = set()
                for field in ("props", "extra_props"):
                    for targets in cached_node.get(field, {}).values():
                        neighbor_qids.update(targets)
                for nqid in neighbor_qids:
                    if nqid in cache_snap:
                        neighbor = cache_snap[nqid]
                        local_nodes[nqid] = neighbor
                        # Also retrieve the targets of the neighbor's extra_props
                        for targets in neighbor.get("extra_props", {}).values():
                            for tqid in targets:
                                if tqid in cache_snap:
                                    local_nodes[tqid] = cache_snap[tqid]
            else:
                # Download node (this issues API calls)
                nodes_q = download_node_multi_ontology(qid, classes, merged_config)
                node_cache.update(nodes_q)
                local_nodes.update(nodes_q)

            # TYPE edges to the ontology's concepts
            type_edges.extend(add_type_edges(qid, classes, ontology_concepts))

        # Build output
        prop_edges = extract_prop_edges(local_nodes)
        all_edges = prop_edges + edges_to_dict_list(type_edges)

        subgraph = {
            "id": ex_id,
            "matched_ontologies": matched_ont_ids,
            "nodes": list(local_nodes.values()),
            "edges": all_edges,
        }

        # Save
        out_file = output_path / f"{ex_id}.json"
        with open(out_file, "w", encoding="utf-8") as fout:
            json.dump(subgraph, fout, ensure_ascii=False)
        
        return (ex_id, True, None)
    
    except Exception as e:
        return (ex_id, False, str(e))


def process_dataset(
    input_file: str,
    ontologies_dir: str,
    output_dir: str,
    cache_file: str = "node_cache.pkl",
    checkpoint_every: int = 100,
    resume: bool = True,
    max_workers: int = 4,
):
    """
    Process the entire dataset and generate a subgraph per question.
    Uses multi-threading to speed things up.

    Args:
        input_file: MetaHotpotQA_with_ontologies.jsonl
        ontologies_dir: ontologies directory
        output_dir: directory where the subgraphs are saved
        cache_file: pickle file for the node cache
        checkpoint_every: save cache every N examples
        resume: if True, skip already-processed examples
        max_workers: number of parallel threads
    """
    input_path = Path(input_file)
    ontologies_path = Path(ontologies_dir)
    output_path = Path(output_dir)
    cache_path = Path(cache_file)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load ontologies
    print("[*] Loading ontologies...")
    all_ontologies = load_all_ontologies(ontologies_path)
    print(f"    Loaded {len(all_ontologies)} ontologies")

    # Load / initialize the thread-safe cache
    if cache_path.exists():
        print(f"[*] Loading cache from {cache_path}...")
        with open(cache_path, "rb") as f:
            initial_cache = pickle.load(f)
        node_cache = ThreadSafeCache(initial_cache)
        print(f"    {len(node_cache)} cached nodes")
    else:
        node_cache = ThreadSafeCache()

    # Find already-processed examples
    done_ids = set()
    if resume:
        done_ids = {
            fn[:-5] for fn in os.listdir(output_path)
            if fn.endswith(".json")
        }
        print(f"[*] Already-processed examples: {len(done_ids)}")

    # Load examples to process
    print("[*] Loading examples...")
    examples_to_process = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                example = json.loads(line)
                if example["id"] not in done_ids:
                    examples_to_process.append(example)
            except json.JSONDecodeError:
                continue
    
    total = len(examples_to_process)
    print(f"[*] Examples to process: {total}")

    if total == 0:
        print("[OK] No examples to process.")
        return

    # Process via ThreadPoolExecutor
    processed = 0
    errors = 0
    start_time = time.time()

    print(f"[*] Starting processing with {max_workers} threads...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(
                build_subgraph_for_example_safe,
                example,
                all_ontologies,
                node_cache,
                output_path,
            ): example["id"]
            for example in examples_to_process
        }
        
        # Process results as they arrive
        for future in as_completed(futures):
            ex_id = futures[future]
            try:
                result_id, success, error_msg = future.result()
                if success:
                    processed += 1
                else:
                    errors += 1
                    print(f"[!] Error {ex_id}: {error_msg}")
            except Exception as e:
                errors += 1
                print(f"[!] Exception {ex_id}: {e}")

            # Progress
            done = processed + errors
            if done % 50 == 0:
                elapsed = time.time() - start_time
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else 0
                print(f"    [{done}/{total}] {rate:.1f}/s, "
                      f"OK: {processed}, ERR: {errors}, "
                      f"ETA: {eta/60:.0f}min, Cache: {len(node_cache)}")

            # Periodic cache checkpoint
            if done % checkpoint_every == 0:
                cache_snapshot = node_cache.get_snapshot()
                with open(cache_path, "wb") as f:
                    pickle.dump(cache_snapshot, f)

    # Final checkpoint
    cache_snapshot = node_cache.get_snapshot()
    with open(cache_path, "wb") as f:
        pickle.dump(cache_snapshot, f)

    elapsed = time.time() - start_time
    print(f"\n[OK] Done!")
    print(f"    Processed: {processed}")
    print(f"    Errors: {errors}")
    print(f"    Time: {elapsed/60:.1f} min")
    print(f"    Rate: {total/elapsed:.1f} examples/s")
    print(f"    Cache: {len(node_cache)} nodes")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build Wikidata subgraphs for MetaHotpotQA")
    parser.add_argument("--workers", "-w", type=int, default=4,
                        help="Number of parallel threads (default: 4)")
    parser.add_argument("--checkpoint", "-c", type=int, default=100,
                        help="Save cache every N examples (default: 100)")
    parser.add_argument(
        "--ontologies-dir",
        default=None,
        help="Ontologies directory (default: ./ontologies)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for the subgraphs (default: ./subgraphs)",
    )
    parser.add_argument(
        "--cache-file",
        default=None,
        help="Path to the cache pickle file (default: ./node_cache.pkl)",
    )
    parser.add_argument(
        "--input",
        default="data/hotpot_matched.jsonl",
        help="Input JSONL from Stage 1b (default: data/hotpot_matched.jsonl)",
    )
    args = parser.parse_args()

    input_dataset = Path(args.input).resolve()
    ontologies_directory = Path(args.ontologies_dir) if args.ontologies_dir else Path("ontologies")
    output_directory = Path(args.output_dir) if args.output_dir else Path("data") / "subgraphs"
    cache_file = Path(args.cache_file) if args.cache_file else Path("data") / "node_cache.pkl"

    print(f"Input:      {input_dataset}")
    print(f"Ontologies: {ontologies_directory}")
    print(f"Output:     {output_directory}")
    print(f"Cache:      {cache_file}")
    print(f"Workers:    {args.workers}")
    print(f"Checkpoint: every {args.checkpoint} examples\n")

    if not input_dataset.exists():
        print(f"[!] ERROR: input not found: {input_dataset}")
        exit(1)
    if not ontologies_directory.exists():
        print(f"[!] ERROR: ontologies not found: {ontologies_directory}")
        exit(1)

    process_dataset(
        input_file=str(input_dataset),
        ontologies_dir=str(ontologies_directory),
        output_dir=str(output_directory),
        cache_file=str(cache_file),
        checkpoint_every=args.checkpoint,
        resume=True,
        max_workers=args.workers,
    )
