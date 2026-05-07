"""
02_bridge_patcher.py

For S0 examples (entity_selection) where the seed nodes are disconnected in
the subgraph (supporting_paths = []), query Wikidata SPARQL to find a 1- or
2-hop path connecting them, patch the subgraph with the new nodes/edges, and
re-annotate supporting_paths.

Pipeline:
  1. Load S0 records with empty supporting_paths from the splits
  2. For each, search Wikidata for a connection (1-hop then 2-hop)
  3. Download label/description for missing intermediate nodes
  4. Patch the subgraph JSON file with the new nodes/edges
  5. Re-annotate supporting_paths in the split record
  6. Save the updated files

Usage:
    python pipeline/4_path_annotation/02_bridge_patcher.py [--splits DIR] [--subgraphs DIR]
                                                            [--output-splits DIR] [--dry-run]
"""

import argparse
import json
import time
import os
import requests
from pathlib import Path

# When invoked as a script, the sibling 01_annotate_paths.py is on sys.path.
# Import its build_graph/bfs_segment helpers via dynamic import (file name
# starts with a digit so a normal import statement won't work).
import importlib.util as _importlib_util
_path_finder_spec = _importlib_util.spec_from_file_location(
    "_path_finder", Path(__file__).parent / "01_annotate_paths.py"
)
_path_finder = _importlib_util.module_from_spec(_path_finder_spec)
_path_finder_spec.loader.exec_module(_path_finder)
build_graph = _path_finder.build_graph
bfs_segment = _path_finder.bfs_segment

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"


def _user_agent() -> str:
    contact = os.environ.get("WIKIDATA_CONTACT", "").strip()
    if not contact:
        raise RuntimeError(
            "WIKIDATA_CONTACT env var is required (set it to a contact email "
            "or URL identifying you, per Wikidata's User-Agent policy).")
    return f"MetaHotpotQA-s0-bridge/1.0 (https://github.com/FValerio96/MetaHotpot; mailto:{contact})"


HEADERS = {"User-Agent": _user_agent()}


# ---------------------------------------------------------------------------
# Wikidata helpers
# ---------------------------------------------------------------------------

def sparql_query(query: str, retries=4) -> list[dict]:
    for attempt in range(retries):
        try:
            r = requests.get(
                SPARQL_ENDPOINT,
                params={"query": query, "format": "json"},
                headers=HEADERS,
                timeout=30,
            )
            if r.status_code == 429:
                time.sleep(10 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()["results"]["bindings"]
        except Exception as e:
            wait = 5 * (attempt + 1)
            print(f"    [SPARQL error attempt {attempt+1}, retry in {wait}s] {e}")
            time.sleep(wait)
    return []


def find_1hop(qid_a: str, qid_b: str) -> list[dict]:
    """
    Search direct A->B and B->A triples in Wikidata in a single query.
    Returns a list of {pid, direction}.
    """
    query = f"""
    SELECT ?p ?dir WHERE {{
      {{
        wd:{qid_a} ?p wd:{qid_b} .
        ?prop wikibase:directClaim ?p .
        BIND("forward" AS ?dir)
      }} UNION {{
        wd:{qid_b} ?p wd:{qid_a} .
        ?prop wikibase:directClaim ?p .
        BIND("inverse" AS ?dir)
      }}
    }} LIMIT 10
    """
    results = sparql_query(query)
    return [
        {
            "pid": r["p"]["value"].split("/")[-1],
            "direction": r.get("dir", {}).get("value", "forward"),
        }
        for r in results
    ]


def find_2hop(qid_a: str, qid_b: str) -> list[dict]:
    """
    Search 2-hop paths in Wikidata: forward (A->X->B) first, then inverse
    (B->X->A). Two lightweight queries instead of one heavy UNION.
    """
    paths = []
    for src, tgt, direction in [(qid_a, qid_b, "forward"), (qid_b, qid_a, "inverse")]:
        query = f"""
        SELECT DISTINCT ?p1 ?x ?p2 WHERE {{
          wd:{src} ?p1 ?x .
          ?x ?p2 wd:{tgt} .
          ?prop1 wikibase:directClaim ?p1 .
          ?prop2 wikibase:directClaim ?p2 .
          FILTER(?x != wd:{src} && ?x != wd:{tgt})
          FILTER(STRSTARTS(STR(?x), "http://www.wikidata.org/entity/Q"))
        }} LIMIT 3
        """
        results = sparql_query(query)
        for r in results:
            paths.append({
                "p1": r["p1"]["value"].split("/")[-1],
                "x_qid": r["x"]["value"].split("/")[-1],
                "p2": r["p2"]["value"].split("/")[-1],
                "direction": direction,
            })
        time.sleep(0.5)
        if paths:
            break  # one direction is enough
    return paths


def fetch_node_label(qid: str) -> dict:
    """Download the label and description of a Wikidata node."""
    query = f"""
    SELECT ?label ?desc WHERE {{
      OPTIONAL {{ wd:{qid} rdfs:label ?label . FILTER(LANG(?label) = "en") }}
      OPTIONAL {{ wd:{qid} schema:description ?desc . FILTER(LANG(?desc) = "en") }}
    }} LIMIT 1
    """
    results = sparql_query(query)
    if results:
        return {
            "qid": qid,
            "label": results[0].get("label", {}).get("value", qid),
            "description": results[0].get("desc", {}).get("value", ""),
            "instance_of": [],
            "subclass_of": [],
            "props": {},
            "extra_props": {},
        }
    return {"qid": qid, "label": qid, "description": "", "instance_of": [],
            "subclass_of": [], "props": {}, "extra_props": {}}


# ---------------------------------------------------------------------------
# Subgraph patching
# ---------------------------------------------------------------------------

def patch_subgraph_1hop(sg: dict, qid_a: str, qid_b: str, pid: str, direction: str) -> bool:
    """
    Add the direct edge between qid_a and qid_b to the subgraph.
    direction='forward': qid_a -> qid_b
    direction='inverse': qid_b -> qid_a
    """
    src, tgt = (qid_a, qid_b) if direction == "forward" else (qid_b, qid_a)
    new_edge = {"source": src, "type": pid, "target": tgt}

    existing = {(e["source"], e["type"], e["target"]) for e in sg.get("edges", [])}
    if (src, pid, tgt) in existing:
        return False

    sg.setdefault("edges", []).append(new_edge)
    return True


def patch_subgraph_2hop(sg: dict, qid_a: str, qid_b: str,
                        p1: str, x_qid: str, p2: str, direction: str,
                        x_node: dict) -> bool:
    """
    Add the intermediate node X and the two edges to the subgraph.
    direction='forward': qid_a -p1-> x -p2-> qid_b
    direction='inverse': qid_b -p1-> x -p2-> qid_a
    """
    if direction == "forward":
        src1, tgt1 = qid_a, x_qid
        src2, tgt2 = x_qid, qid_b
    else:
        src1, tgt1 = qid_b, x_qid
        src2, tgt2 = x_qid, qid_a

    existing_qids = {n["qid"] for n in sg.get("nodes", [])}
    if x_qid not in existing_qids:
        sg.setdefault("nodes", []).append(x_node)

    existing_edges = {(e["source"], e["type"], e["target"]) for e in sg.get("edges", [])}
    changed = False
    for src, pid, tgt in [(src1, p1, tgt1), (src2, p2, tgt2)]:
        if (src, pid, tgt) not in existing_edges:
            sg.setdefault("edges", []).append({"source": src, "type": pid, "target": tgt})
            changed = True

    return changed


# ---------------------------------------------------------------------------
# Path annotation update
# ---------------------------------------------------------------------------

def build_supporting_path_1hop(src: str, pid: str, tgt: str) -> dict:
    return {"length": 1, "triples": [[src, pid, tgt]]}


def build_supporting_path_2hop(src1: str, p1: str, x: str,
                                p2: str, tgt2: str) -> dict:
    return {"length": 2, "triples": [[src1, p1, x], [x, p2, tgt2]]}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_split(split_path: Path, subgraphs_dir: Path,
                  output_split_path: Path, output_subgraphs_dir: Path,
                  dry_run: bool = False) -> dict:

    records = []
    with open(split_path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    # Preflight: report the source state to avoid running the patcher on a
    # split without S0 annotations (a scenario that produces degraded output)
    s0_records = [r for r in records if r.get("difficulty") == "entity_selection"]
    s0_already_annotated = sum(1 for r in s0_records if r.get("supporting_paths"))
    s0_to_patch = len(s0_records) - s0_already_annotated
    print(f"  Preflight: {len(s0_records)} S0 in source | "
          f"already with path (preserved): {s0_already_annotated} | "
          f"to try via SPARQL: {s0_to_patch}")
    if s0_already_annotated == 0 and len(s0_records) > 0:
        print(f"  WARNING: no S0 has supporting_paths in the source — "
              f"check that you are using the canonical split (e.g. splits_final), "
              f"not the pre-annotation version.")

    stats = {
        "total_s0": 0,
        "already_has_path": 0,
        "patched_1hop": 0,
        "patched_2hop": 0,
        "not_found": 0,
    }

    output_subgraphs_dir.mkdir(parents=True, exist_ok=True)

    # Resume: load IDs already processed from the partial output
    processed_ids: dict[str, dict] = {}
    if output_split_path.exists():
        with open(output_split_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                processed_ids[r["id"]] = r
        print(f"  Resume: {len(processed_ids)} records already processed, resuming.")

    updated_records = []

    # Open output in append mode for resume
    out_mode = "a" if processed_ids else "w"
    out_f = open(output_split_path, out_mode, encoding="utf-8") if not dry_run else None

    try:
        for i, record in enumerate(records):

            # Resume: use the already-processed record
            if record["id"] in processed_ids:
                updated_records.append(processed_ids[record["id"]])
                if record.get("difficulty") == "entity_selection":
                    stats["total_s0"] += 1
                    if processed_ids[record["id"]].get("supporting_paths"):
                        stats["already_has_path"] += 1
                    else:
                        stats["not_found"] += 1
                continue

            if record.get("difficulty") != "entity_selection":
                updated_records.append(record)
                if out_f:
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_f.flush()
                continue

            stats["total_s0"] += 1

            # Already annotated in the source split
            if record.get("supporting_paths"):
                stats["already_has_path"] += 1
                updated_records.append(record)
                if out_f:
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_f.flush()
                continue

            answer_qid = record["answer_node_qid"]
            other_seeds = [q for q in record.get("entity_qids", []) if q != answer_qid]
            if not other_seeds:
                updated_records.append(record)
                if out_f:
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_f.flush()
                continue

            seed = other_seeds[0]

            sg_path = subgraphs_dir / f"{record['id']}.json"
            if not sg_path.exists():
                updated_records.append(record)
                if out_f:
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_f.flush()
                continue

            with open(sg_path, encoding="utf-8") as f:
                sg = json.load(f)

            print(f"[{i+1}] {record['question'][:70]}")
            print(f"  seed={seed}, answer={answer_qid}")

            found = False

            # --- 1-hop ---
            triples_1hop = find_1hop(seed, answer_qid)
            time.sleep(0.5)

            if triples_1hop:
                best = triples_1hop[0]
                pid, direction = best["pid"], best["direction"]
                src, tgt = (seed, answer_qid) if direction == "forward" else (answer_qid, seed)
                print(f"  1-hop found: {src} -[{pid}]-> {tgt}")

                if not dry_run:
                    patch_subgraph_1hop(sg, seed, answer_qid, pid, direction)
                    record["supporting_paths"] = [build_supporting_path_1hop(src, pid, tgt)]

                stats["patched_1hop"] += 1
                found = True

            else:
                # --- 2-hop ---
                paths_2hop = find_2hop(seed, answer_qid)

                if paths_2hop:
                    best = paths_2hop[0]
                    p1, x_qid, p2, direction = best["p1"], best["x_qid"], best["p2"], best["direction"]
                    print(f"  2-hop found via {x_qid}: {p1} -> {x_qid} -> {p2}")

                    x_node = fetch_node_label(x_qid)
                    time.sleep(0.5)

                    if direction == "forward":
                        src1, tgt2 = seed, answer_qid
                    else:
                        src1, tgt2 = answer_qid, seed

                    if not dry_run:
                        patch_subgraph_2hop(sg, seed, answer_qid, p1, x_qid, p2, direction, x_node)
                        record["supporting_paths"] = [
                            build_supporting_path_2hop(src1, p1, x_qid, p2, tgt2)
                        ]

                    stats["patched_2hop"] += 1
                    found = True

            if not found:
                stats["not_found"] += 1
                print(f"  Not found")

            # Save the patched subgraph
            if not dry_run and found:
                out_sg_path = output_subgraphs_dir / f"{record['id']}.json"
                with open(out_sg_path, "w", encoding="utf-8") as f:
                    json.dump(sg, f, ensure_ascii=False)

            updated_records.append(record)
            if out_f:
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
            print()

    finally:
        if out_f:
            out_f.close()

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", default="data/splits_internal",
                        help="Source splits directory (must contain S0 "
                             "annotations from path_finder: S0 records with "
                             "supporting_paths already populated are preserved, "
                             "empty ones are passed to SPARQL)")
    parser.add_argument("--subgraphs", default="data/subgraphs",
                        help="Subgraphs directory (output of Stage 2)")
    parser.add_argument("--output-splits", default="data/splits_patched",
                        help="Output directory for the updated splits")
    parser.add_argument("--output-subgraphs", default="data/subgraphs_patched",
                        help="Output directory for the patched subgraphs (input of 2/04_merge_final)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print only, do not write anything")
    parser.add_argument("--split", default=None,
                        help="Process only this split (e.g. train, dev, test)")
    args = parser.parse_args()

    splits_dir = Path(args.splits)
    subgraphs_dir = Path(args.subgraphs)
    output_splits_dir = Path(args.output_splits)
    output_subgraphs_dir = Path(args.output_subgraphs)

    output_splits_dir.mkdir(parents=True, exist_ok=True)

    split_files = (
        [splits_dir / f"{args.split}.jsonl"] if args.split
        else sorted(splits_dir.glob("*.jsonl"))
    )

    total_stats = {}
    for split_path in split_files:
        print(f"\n{'='*60}")
        print(f"Processing split: {split_path.name}")
        print(f"{'='*60}\n")

        stats = process_split(
            split_path=split_path,
            subgraphs_dir=subgraphs_dir,
            output_split_path=output_splits_dir / split_path.name,
            output_subgraphs_dir=output_subgraphs_dir,
            dry_run=args.dry_run,
        )
        total_stats[split_path.stem] = stats

        print(f"\nSplit {split_path.stem} summary:")
        print(f"  Total S0:           {stats['total_s0']}")
        print(f"  Already with path:  {stats['already_has_path']}")
        print(f"  Patched 1-hop:      {stats['patched_1hop']}")
        print(f"  Patched 2-hop:      {stats['patched_2hop']}")
        print(f"  Not found:          {stats['not_found']}")

    if not args.dry_run:
        print(f"\nUpdated splits -> {output_splits_dir}")
        print(f"Patched subgraphs -> {output_subgraphs_dir}")


if __name__ == "__main__":
    main()
