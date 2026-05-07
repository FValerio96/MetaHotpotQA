"""
01_annotate_paths.py — Reasoning-path annotation on the extended MetaHotpotQA KG.

================================================================================
  BACKGROUND: BRIDGE vs COMPARISON in HotpotQA
================================================================================

  HotpotQA (Yang et al., 2018) defines two fundamental types of multi-hop
  question, differing in the structure of reasoning required:

  BRIDGE — chain traversal reasoning
      The answer is found by walking a chain of Wikipedia pages.
      A "bridge" page connects the question to the answer: the first page
      introduces the intermediate entity, the second page contains the answer.
      Example: "What is the hometown of the director of Inception?"
        -> Inception (page 1) -> Christopher Nolan (bridge entity) -> London (answer)
      The model must navigate the chain; each step is a supporting page.
      On the KG this becomes graph traversal: follow edges from a seed node
      up to the answer node.

  COMPARISON — parallel property comparison
      The answer is one of the two entities mentioned in the question.
      No traversal: the two pages are read in parallel and a shared property
      is compared to select the right one.
      Example: "Were Scott Derrickson and Ed Wood of the same nationality?"
        -> read Scott Derrickson's page (American)
        -> read Ed Wood's page (American)
        -> compare nationalities -> answer: yes
      On the KG this becomes property lookup: for each seed entity, collect
      the 1-hop properties and compare them. There is no path to follow.

  CONSEQUENCE FOR THE PATH FINDER:
      The two classes require different strategies:
        - Bridge -> BFS to find the path connecting the seeds to the answer
        - Comparison -> dump of 1-hop properties of both entities
      The discriminating property (e.g. P577 for "which film is older?") must
      be present in the subgraph for the comparison to be KG-solvable.
      When missing, the model has to fall back on parametric knowledge — this
      tells us which properties should be added to the ontologies.

================================================================================
  STRATEGIES AND DIFFICULTY LEVELS
================================================================================

  The strategies do not represent a quality hierarchy but a difficulty
  taxonomy for KGQA systems. The dataset is valid in all cases: what changes
  is the type of reasoning required from the system to answer.

  BRIDGE QUESTIONS — cascading strategy:

    Strategy 0 — Entity Selection  [difficulty: "entity_selection"]
        answer_node_qid is already present in the question's `entities` field,
        i.e. it is one of the subgraph's seed nodes. This happens when HotpotQA
        includes the answer's Wikipedia page among the supporting facts.
        The KGQA system still has to perform entity linking and graph traversal:
        the answer node is a seed, but the reasoning requires following the
        chain of relations between the seed nodes to figure out WHICH seed is
        the answer (and why).
        For example: book --[P50 (author)]--> Ian Stevenson. Without following
        that edge one cannot establish that Ian Stevenson is the author, and
        therefore the answer.
        path_finder attempts BFS between the seed nodes; if connected,
        supporting_paths contains the found path. If disconnected in the
        subgraph, the bridge patcher queries Wikidata for 1-2 hop connections.
        path_found=True.

    Strategy 1 — Multi-hop Waypoint Traversal  [difficulty: "traversal"]
        The path must mandatorily transit through ALL entity QIDs as
        intermediate nodes before reaching answer_node_qid. Reflects HotpotQA's
        canonical bridge reasoning: every supporting page is a mandatory step
        in the reasoning chain.
        All permutations of the intermediates are tried (capped at
        MAX_INTERMEDIATE_FOR_PERM to avoid combinatorial blow-up).
        Max hops per segment: --max-hops-segment (default 2).
        Requires the system to concatenate several hops through specific
        waypoints.
        path_found=True.

    Strategy 2 — Direct Graph Traversal  [difficulty: "traversal"]
        If S1 fails, BFS without waypoint constraints from any seed entity to
        answer_node_qid. A structural path exists in the graph but does not
        necessarily go through every seed. The system must navigate the graph
        starting from at least one seed to reach the answer.
        Max total hops: --max-hops-direct (default 4).
        path_found=True.

    Strategy 3 — No Structural Path  [difficulty: "no_path"]
        answer_node_qid exists in the subgraph (verified by answer_finder.py)
        but is not structurally reachable from the seed entities within the
        hop limits. The answer is in the KG but the graph is locally
        disconnected for that question. path_found=False.

  COMPARISON QUESTIONS  [difficulty: "property_comparison"]
        Collects the 1-hop properties of each seed entity (comparison_triples).
        The answer is already one of the seeds (e.g. "between A and B, which
        has the higher property X?" -> answer is A or B). The system must
        compare a shared property of the two entities to select the correct
        one. There is no path to find: the reasoning is lookup + comparison.
        path_found=True if at least one seed has properties in the subgraph.

        NOTE ON MISSING PROPERTIES: the subgraph is optimized for structural
        traversal, not for quantitative comparison. Properties such as P577
        (publication date), P625 (geographic coordinates), or member counts
        are often absent. When missing, the model must fall back on
        parametric knowledge. Every comparison question that is not KG-
        solvable points to a property that should be added to the ontology
        for that entity class — the dataset acts as a specification for KG
        extension.

  difficulty FIELD:
        Each record includes a "difficulty" field that indicates the type of
        reasoning required:
          "entity_selection"   — bridge S0: identify answer among seed entities
          "traversal"          — bridge S1/S2: BFS-based graph traversal to answer
          "no_path"            — bridge S3 / comparison not found
          "property_comparison"— comparison found

================================================================================
  PIPELINE
================================================================================

  For each question in found.jsonl (extended):
    1. Load the corresponding subgraph from subgraphs_extended/
    2. Build a directed graph with explicit edges + nodes' extra_props
    3. Apply the appropriate strategy (bridge or comparison)
    4. Save path_annotations/path_annotations.jsonl + summary.json

Usage:
    python 01_annotate_paths.py [--found PATH] [--subgraphs DIR] [--output DIR]
                                [--max-hops-segment N] [--max-hops-direct N]
                                [--max-paths N]

Output for a bridge question:
    {
      "id": "...",
      "question": "...",
      "answer": "...",
      "answer_node_qid": "Q...",
      "entity_qids": ["Q...", "Q..."],
      "reasoning_type": "bridge",
      "match_tier": 1,
      "path_found": true,
      "strategy_used": 1,          // 0=entity_selection, 1=waypoint, 2=direct, 3=no_path
      "difficulty": "traversal",
      "supporting_paths": [        // path between seed nodes (may be empty only if S0 disconnected)
        {"length": 2, "triples": [["Qa", "P_rel", "Qb"], ["Qb", "P_rel2", "Qans"]]}
      ]
    }

Output for a comparison question:
    {
      "reasoning_type": "comparison",
      "path_found": true,
      "strategy_used": 1,
      "difficulty": "property_comparison",
      "comparison_triples": {
        "Q1": [["Q1", "P571", "1844"]],
        "Q2": [["Q2", "P571", "1989"]]
      }
    }
"""

import argparse
import json
import logging
from collections import defaultdict, deque
from itertools import permutations
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[path_finder] %(message)s")
log = logging.getLogger(__name__)

# Beyond this number of intermediate entities, Strategy 1 is skipped to avoid
# combinatorial blow-up (n! permutations). With max=3: at most 6 permutations.
MAX_INTERMEDIATE_FOR_PERM = 3


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph(subgraph: dict) -> dict[str, list[tuple[str, str]]]:
    """
    Build an adjacency-dict graph: graph[src] = [(rel, tgt), ...]
    Includes the subgraph's explicit edges and the nodes' extra_props
    (added by the concepts_to_extend step in the subgraph builder).
    """
    graph: dict[str, list] = defaultdict(list)

    for edge in subgraph.get("edges", []):
        src = edge.get("source", "").strip()
        rel = edge.get("type", "").strip()
        tgt = edge.get("target", "").strip()
        if src and rel and tgt:
            graph[src].append((rel, tgt))

    for node in subgraph.get("nodes", []):
        qid = (node.get("qid") or "").strip()
        if not qid:
            continue
        for prop, values in node.get("extra_props", {}).items():
            for val in values:
                if isinstance(val, str) and val.strip():
                    graph[qid].append((prop, val.strip()))

    return graph


# ---------------------------------------------------------------------------
# BFS with hop limit
# ---------------------------------------------------------------------------

def bfs_segment(
    graph: dict,
    source: str,
    target: str,
    max_hops: int,
    max_paths: int = 20,
) -> list[list[tuple]]:
    """
    Directed BFS from source to target with a max_hops limit.
    Return all found paths (up to max_paths), each as a list of triples
    (src, rel, tgt). Uses per-path visited set to avoid cycles.
    """
    if source == target:
        return []

    found: list[list[tuple]] = []
    # Each queue element: (current_node, accumulated_path, visited_nodes)
    queue: deque = deque([(source, [], frozenset([source]))])

    while queue and len(found) < max_paths:
        node, path, visited = queue.popleft()

        if len(path) >= max_hops:
            continue

        for rel, neighbor in graph.get(node, []):
            if neighbor in visited:
                continue
            triple = (node, rel, neighbor)
            new_path = path + [triple]

            if neighbor == target:
                found.append(new_path)
                if len(found) >= max_paths:
                    break
            else:
                queue.append((neighbor, new_path, visited | {neighbor}))

    return found


# ---------------------------------------------------------------------------
# Bridge strategy
# ---------------------------------------------------------------------------

def find_paths_bridge(
    graph: dict,
    entity_qids: list[str],
    answer_qid: str,
    max_hops_segment: int = 2,
    max_hops_direct: int = 4,
    max_paths: int = 20,
) -> tuple[int, list]:
    """
    Search for reasoning paths for bridge questions with a cascading strategy.

    Strategy 1: paths that mandatorily transit through all entities as
                intermediate waypoints (respects the HotpotQA structure).
    Strategy 2: direct BFS from any seed entity to answer_qid.
    Strategy 3: no path found.

    Returns:
        (strategy_used, paths) — paths is a list of lists of triples.
    """
    # --- Strategy 0: the answer is already a seed node ---
    if answer_qid in entity_qids:
        # seed nodes - answer node
        other_seeds = [q for q in entity_qids if q != answer_qid]

        if not other_seeds:
            return 0, []

        # Option A: sequential path preserving the HotpotQA order
        waypoints = other_seeds + [answer_qid]
        full_path: list[tuple] = []
        valid = True
        prev = waypoints[0]
        for wp in waypoints[1:]:
            segs = bfs_segment(graph, prev, wp, max_hops_segment, max_paths=5)
            if not segs:
                valid = False
                break
            full_path.extend(min(segs, key=len))
            prev = wp

        if valid and full_path:
            return 0, [full_path]

        # Option B: fallback direct BFS from each seed to the answer
        s0_paths = []
        for seed in other_seeds:
            paths = bfs_segment(graph, seed, answer_qid, max_hops_segment, max_paths=5)
            s0_paths.extend(paths)

        if s0_paths:
            seen_paths: set = set()
            unique = []
            for p in sorted(s0_paths, key=len):
                key = tuple(p)
                if key not in seen_paths:
                    seen_paths.add(key)
                    unique.append(p)
            return 0, unique[:max_paths]

        # No structural path found, but the answer is still a seed
        return 0, []

    # --- Strategy 1: waypoint-constrained ---
    intermediates = [q for q in entity_qids if q != answer_qid]
    if intermediates and len(intermediates) <= MAX_INTERMEDIATE_FOR_PERM:
        s1_paths = []
        for perm in permutations(intermediates):
            waypoints = list(perm) + [answer_qid]
            full_path: list[tuple] = []
            valid = True

            prev = waypoints[0]
            for wp in waypoints[1:]:
                segs = bfs_segment(graph, prev, wp, max_hops_segment, max_paths=5)
                if not segs:
                    valid = False
                    break
                # Take the shortest path for this segment
                full_path.extend(min(segs, key=len))
                prev = wp

            if valid and full_path:
                s1_paths.append(full_path)

        if s1_paths:
            # Deduplicate and sort by length
            seen_paths = set()
            unique = []
            for p in sorted(s1_paths, key=len):
                key = tuple(p)
                if key not in seen_paths:
                    seen_paths.add(key)
                    unique.append(p)
            return 1, unique[:max_paths]

    # --- Strategy 2: direct BFS ---
    s2_paths = []
    for eq in entity_qids:
        paths = bfs_segment(graph, eq, answer_qid, max_hops_direct, max_paths)
        s2_paths.extend(paths)

    if s2_paths:
        seen_paths = set()
        unique = []
        for p in sorted(s2_paths, key=len):
            key = tuple(p)
            if key not in seen_paths:
                seen_paths.add(key)
                unique.append(p)
        return 2, unique[:max_paths]

    # --- Strategy 3: no path ---
    return 3, []


# ---------------------------------------------------------------------------
# Comparison strategy
# ---------------------------------------------------------------------------

def find_paths_comparison(
    graph: dict,
    entity_qids: list[str],
    answer_qid: str,
) -> tuple[bool, dict]:
    """
    For comparison questions: collect the 1-hop properties of each seed
    entity. Each entity has a list of (qid, prop, value) triples.
    answer_qid is already one of the seed entities — no chain path needed.
    """
    comparison_triples: dict[str, list] = {}
    for qid in entity_qids:
        triples = [(qid, rel, tgt) for rel, tgt in graph.get(qid, [])]
        comparison_triples[qid] = triples

    path_found = any(len(v) > 0 for v in comparison_triples.values())
    return path_found, comparison_triples


# ---------------------------------------------------------------------------
# Per-question processing
# ---------------------------------------------------------------------------

def process_record(
    record: dict,
    subgraph: dict,
    max_hops_segment: int,
    max_hops_direct: int,
    max_paths: int,
) -> dict:
    """
    Process a question and return the path-annotated result.
    """
    # Deduplicate entity_qids preserving order of appearance
    seen: set[str] = set()
    entity_qids: list[str] = []
    for e in record.get("entities", []):
        qid = (e.get("qid") or "").strip()
        if qid and qid not in seen:
            seen.add(qid)
            entity_qids.append(qid)

    answer_qid = record["answer_node_qid"]
    reasoning_type = record.get("reasoning_type", "bridge")

    graph = build_graph(subgraph)

    result: dict = {
        "id": record["id"],
        "question": record["question"],
        "answer": record["answer"],
        "answer_node_qid": answer_qid,
        "entity_qids": entity_qids,
        "reasoning_type": reasoning_type,
        "match_tier": record.get("match_tier"),
    }

    _bridge_difficulty = {
        0: "entity_selection",
        1: "traversal",
        2: "traversal",
        3: "no_path",
    }

    if reasoning_type == "comparison":
        path_found, comparison_triples = find_paths_comparison(
            graph, entity_qids, answer_qid
        )
        result["path_found"] = path_found
        result["strategy_used"] = 1 if path_found else 3
        result["difficulty"] = "property_comparison" if path_found else "no_path"
        result["comparison_triples"] = {
            qid: [list(t) for t in triples]
            for qid, triples in comparison_triples.items()
        }
    else:
        strategy, paths = find_paths_bridge(
            graph, entity_qids, answer_qid,
            max_hops_segment, max_hops_direct, max_paths,
        )
        result["path_found"] = strategy < 3  # 0, 1, 2 = found; 3 = not found
        result["strategy_used"] = strategy
        result["difficulty"] = _bridge_difficulty[strategy]
        result["supporting_paths"] = [
            {"length": len(p), "triples": [list(t) for t in p]}
            for p in paths
        ]

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reasoning-path annotation for MetaHotpotQA (extended KG)"
    )
    parser.add_argument(
        "--found",
        default="data/answer_search/found.jsonl",
        help="Path to found.jsonl (output of Stage 3). Default: %(default)s",
    )
    parser.add_argument(
        "--subgraphs",
        default="data/subgraphs",
        help="Subgraphs directory (output of Stage 2). Default: %(default)s",
    )
    parser.add_argument(
        "--output",
        default="data/path_annotations",
        help="Output directory. Default: %(default)s",
    )
    parser.add_argument(
        "--max-hops-segment",
        type=int,
        default=2,
        help="Max hops per segment in Strategy 1 (default: 2)",
    )
    parser.add_argument(
        "--max-hops-direct",
        type=int,
        default=4,
        help="Max total hops in Strategy 2 (default: 4)",
    )
    parser.add_argument(
        "--max-paths",
        type=int,
        default=20,
        help="Max paths saved per question (default: 20)",
    )
    args = parser.parse_args()

    found_path = Path(args.found)
    sg_dir = Path(args.subgraphs)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load found.jsonl deduplicating by ID (keeping the first occurrence) ---
    log.info(f"Loading {found_path}")
    records: dict[str, dict] = {}
    duplicates = 0
    with open(found_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["id"] in records:
                duplicates += 1
            else:
                records[r["id"]] = r
    log.info(f"Unique questions: {len(records)} ({duplicates} duplicates dropped)")

    # --- Processing ---
    out_path = output_dir / "path_annotations.jsonl"
    stats = {
        "total": 0,
        "bridge": {"s0": 0, "s1": 0, "s2": 0, "not_found": 0},
        "comparison": {"found": 0, "not_found": 0},
        "subgraph_missing": 0,
    }

    with open(out_path, "w", encoding="utf-8") as out_f:
        for i, (qid, record) in enumerate(records.items(), 1):
            sg_path = sg_dir / f"{qid}.json"

            if not sg_path.exists():
                stats["subgraph_missing"] += 1
                continue

            with open(sg_path, encoding="utf-8") as f:
                subgraph = json.load(f)

            result = process_record(
                record, subgraph,
                args.max_hops_segment,
                args.max_hops_direct,
                args.max_paths,
            )
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")

            # Update statistics
            stats["total"] += 1
            rt = result["reasoning_type"]
            if rt == "comparison":
                key = "found" if result["path_found"] else "not_found"
                stats["comparison"][key] += 1
            else:
                s = result["strategy_used"]
                if s == 0:
                    stats["bridge"]["s0"] += 1
                elif s == 1:
                    stats["bridge"]["s1"] += 1
                elif s == 2:
                    stats["bridge"]["s2"] += 1
                else:
                    stats["bridge"]["not_found"] += 1

            if i % 500 == 0:
                log.info(f"  {i}/{len(records)} processed...")

    # --- Summary ---
    bridge_total = stats["bridge"]["s0"] + stats["bridge"]["s1"] + stats["bridge"]["s2"] + stats["bridge"]["not_found"]
    comp_total = stats["comparison"]["found"] + stats["comparison"]["not_found"]
    bridge_found = stats["bridge"]["s0"] + stats["bridge"]["s1"] + stats["bridge"]["s2"]
    bridge_waypoint = stats["bridge"]["s1"]
    summary = {
        **stats,
        "bridge_path_found_pct": round(100 * bridge_found / bridge_total, 1) if bridge_total else 0,
        "bridge_waypoint_traversal_pct": round(100 * bridge_waypoint / bridge_total, 1) if bridge_total else 0,
        "comparison_found_pct": round(
            100 * stats["comparison"]["found"] / comp_total, 1
        ) if comp_total else 0,
        "params": {
            "max_hops_segment": args.max_hops_segment,
            "max_hops_direct": args.max_hops_direct,
            "max_paths": args.max_paths,
        },
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    log.info(f"Annotations -> {out_path}")
    log.info(f"Summary -> {summary_path}")
    log.info(
        f"Bridge  — S0(entity_sel): {stats['bridge']['s0']} | S1(waypoint): {stats['bridge']['s1']} | "
        f"S2(direct): {stats['bridge']['s2']} | no_path: {stats['bridge']['not_found']} | "
        f"path found: {summary['bridge_path_found_pct']}% | waypoint: {summary['bridge_waypoint_traversal_pct']}%"
    )
    log.info(
        f"Comparison — found: {stats['comparison']['found']} | "
        f"not found: {stats['comparison']['not_found']} | "
        f"found: {summary['comparison_found_pct']}%"
    )


if __name__ == "__main__":
    main()
