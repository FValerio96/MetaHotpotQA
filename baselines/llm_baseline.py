"""
llm_baseline.py — LLM baseline (via Ollama) for MetaHotpotQA.

Two modes:
closed_book  — Question only. The LLM answers from parametric knowledge,
                with no KG or text context. Measures how solvable the
                dataset is without access to the graph.

oracle_kg    — Question + KG context in oracle format. The context depends
                on the record's difficulty:
                - traversal:
                    triples of the shortest supporting_path.
                - entity_selection with supporting_paths:
                    same format as traversal.
                - entity_selection without path (disconnected seeds):
                    full 1-hop ontology-driven neighborhood of each seed
                    (outgoing + incoming, no cap).
                - property_comparison:
                    full 1-hop ontology-driven neighborhood of each seed
                    (same format as the S0-no-path fallback).
                Oracle setting (perfect retrieval) for the upper bound.

Human-readable labels:
Wikidata PIDs (P161, P57, ...) and class QIDs (Q11424, Q5, ...) are replaced
with human labels from the 10 ontologies (ontologies/*.json), loaded via
utils.ontology_utils.load_ontology_labels(). When a PID/QID is not in the
ontologies, the raw code is used as a fallback.

Output JSONL — one record per question:
    {"id": ..., "pred_answer": ..., "pred_qid": null,
    "mode": "closed_book" | "oracle_kg",
    "difficulty": ..., "reasoning_type": ...}

Usage (from any CWD; defaults resolve relative to the project):
    python -m baselines.llm_baseline --mode closed_book --output PATH
    python -m baselines.llm_baseline --mode oracle_kg   --output PATH

Then evaluate with:
    python -m baselines.evaluate --predictions PATH --split PATH
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import requests

# Absolute import: requires the module to be invoked as `python -m baselines.llm_baseline`
# from the project root, or for the root to be on PYTHONPATH.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from utils.ontology_utils import load_ontology_labels  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[llm_baseline] %(message)s")
log = logging.getLogger(__name__)

DEFAULT_OLLAMA_URL   = "http://127.0.0.1:38472"
DEFAULT_OLLAMA_MODEL = "gemma2:12b"
DEFAULT_SUBGRAPHS    = str(_PROJECT_ROOT / "subgraph_dataset" / "test")
DEFAULT_SPLIT        = str(_PROJECT_ROOT / "published_splits" / "test.jsonl")


# ---------------------------------------------------------------------------
# KG context serialization
# ---------------------------------------------------------------------------

def _load_subgraph(subgraph_dir: Path, record_id: str,
                fallback_dir: Path | None = None) -> dict:
    path = subgraph_dir / f"{record_id}.json"
    if not path.exists() and fallback_dir:
        path = fallback_dir / f"{record_id}.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _qid_to_label(subgraph: dict, class_labels: dict[str, str],
                entity_labels: dict[str, str] | None = None) -> dict[str, str]:
    """
    Build a QID -> label map for the subgraph.

    Priority (lowest to highest):
    1. ``entity_labels``: Wikidata labels fetched (cache) — fallback for
        entities reached as ``extra_props target`` without label in the subgraph.
    2. ``class_labels``: class labels from the ontologies (for class-QID
        targets of TYPE that are often not in nodes).
    3. explicit label in the subgraph nodes (local, domain-specific).

    If none of the three is available, the caller will use the raw QID.
    """
    mapping: dict[str, str] = {}
    if entity_labels:
        mapping.update(entity_labels)
    mapping.update(class_labels)  # class wins over entity (ontology precedence)
    for node in subgraph.get("nodes", []):
        qid = (node.get("qid") or "").strip()
        label = (node.get("label") or "").strip()
        if qid and label:
            mapping[qid] = label  # node label wins over everything
    return mapping


def _serialize_triple(triple: list, label_map: dict[str, str],
                    pid_labels: dict[str, str]) -> str:
    """
    Serialize a triple (src, rel, tgt) in a readable format.

    - src and tgt are resolved via label_map (node labels + class labels fallback)
    - rel is resolved via pid_labels (property label from the ontologies). If the
    relation is "TYPE" (the implicit instance_of edge of the subgraphs) we print
    it as "instance of" for semantic uniformity.
    """
    src, rel, tgt = triple
    src_label = label_map.get(src, src)
    tgt_label = label_map.get(tgt, tgt)
    if rel == "TYPE":
        rel_label = "instance of"
    else:
        rel_label = pid_labels.get(rel, rel)
    return f"({src_label}) --[{rel_label}]--> ({tgt_label})"


def _format_1hop_neighbourhood(called_nodes: list[str], subgraph: dict,
                            label_map: dict[str, str],
                            pid_labels: dict[str, str]) -> list[str]:
    """
    Format the 1-hop neighborhood (outgoing + incoming, no cap) for each
    called node. This is the single format used across all difficulties.

    Deduplication (two levels):
    1. An edge between two called_nodes is shown only as outgoing of its
        source (not repeated as incoming of the target).
    2. Identical edges at the (source, type, target) level are emitted only
        once — subgraphs may contain duplicate triples due to the 1-hop
        expansion merging identical triples from different sources.

    Iteration order follows ``called_nodes``: for cases with a path
    (traversal and S0-with-path), passing the nodes in the order
    ``seed -> ... -> answer`` makes the prompt naturally readable from the
    start of the reasoning chain.
    """
    edges = subgraph.get("edges", [])
    called_set = set(called_nodes)
    emitted: set[tuple[str, str, str]] = set()

    def _rel_label(rel: str) -> str:
        return "instance of" if rel == "TYPE" else pid_labels.get(rel, rel)

    lines: list[str] = []
    for qid in called_nodes:
        outs = [e for e in edges if e.get("source") == qid]
        # For incoming: skip edges whose source is another called_node
        # (it will already have been emitted as outgoing of that node).
        ins = [e for e in edges
            if e.get("target") == qid and e.get("source") not in called_set]
        if not (outs or ins):
            continue
        entity_label = label_map.get(qid, qid)
        header_emitted = False
        for e in outs:
            key = (e["source"], e["type"], e["target"])
            if key in emitted:
                continue
            emitted.add(key)
            if not header_emitted:
                lines.append(f"Entity: {entity_label}")
                header_emitted = True
            tgt = label_map.get(e["target"], e["target"])
            lines.append(f"  --[{_rel_label(e['type'])}]--> {tgt}")
        for e in ins:
            key = (e["source"], e["type"], e["target"])
            if key in emitted:
                continue
            emitted.add(key)
            if not header_emitted:
                lines.append(f"Entity: {entity_label}")
                header_emitted = True
            src = label_map.get(e["source"], e["source"])
            lines.append(f"  <--[{_rel_label(e['type'])}]-- {src}")
    return lines


def _path_nodes(path_triples: list[list[str]]) -> list[str]:
    """
    Extract the unique QIDs from a supporting_path's triples preserving the
    order of appearance (seed -> intermediate(s) -> answer for traversal and
    S0-with-path).
    """
    nodes: list[str] = []
    for src, _rel, tgt in path_triples:
        if src not in nodes:
            nodes.append(src)
        if tgt not in nodes:
            nodes.append(tgt)
    return nodes


def build_kg_context(record: dict, subgraph: dict) -> str:
    """
    Build the textual KG context for the oracle_kg mode.

    Unified logic: for each difficulty we determine a set of *called nodes*
    — the nodes relevant to the question — and serialize their 1-hop
    neighborhood (outgoing + incoming) via ``_format_1hop_neighbourhood``.
    Edges between called_nodes (e.g. those of the reasoning chain for
    traversal and S0-with-path) are shown only once, as outgoing of their
    source.

    difficulty -> called_nodes mapping:
    - ``traversal``: all nodes of the shortest supporting_path
        (seed + any intermediates + answer).
    - ``entity_selection`` with supporting_paths: nodes of the path
        (both seeds + any intermediates). The answer is already a seed.
    - ``entity_selection`` without supporting_paths: only the seeds
        (entity_qids), no recoverable intermediate.
    - ``property_comparison``: the two seeds to compare (entity_qids).
    - ``no_path``: no context (the record should already be filtered out
        at split level, but we return an empty string for safety).

    PIDs and class QIDs are replaced with human labels from the ontologies
    when available (see ``_format_1hop_neighbourhood``).
    """
    labels = load_ontology_labels()
    pid_labels = labels["pid_labels"]
    class_labels = labels["class_labels"]
    entity_labels = labels.get("entity_labels", {})
    label_map = _qid_to_label(subgraph, class_labels, entity_labels)
    difficulty = record.get("difficulty", "")

    called_nodes: list[str] = []

    if difficulty == "traversal":
        paths = record.get("supporting_paths", [])
        if paths:
            best = min(paths, key=lambda p: p["length"])
            called_nodes = _path_nodes(best["triples"])

    elif difficulty == "entity_selection":
        paths = record.get("supporting_paths", [])
        if paths:
            best = min(paths, key=lambda p: p["length"])
            called_nodes = _path_nodes(best["triples"])
        else:
            called_nodes = list(record.get("entity_qids", []))

    elif difficulty == "property_comparison":
        called_nodes = list(record.get("entity_qids", []))

    # no_path or unknown difficulty: called_nodes stays empty

    if not called_nodes:
        return ""

    lines = _format_1hop_neighbourhood(
        called_nodes, subgraph, label_map, pid_labels
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a precise question-answering assistant. "
    "Answer each question with only the answer, as briefly as possible. "
    "Do not explain, do not add preamble, do not repeat the question."
)

# Few-shot for closed-book: two factual questions with concise answers.
_FEW_SHOT_CLOSED_BOOK = [
    {"role": "user",      "content": "What is the capital of France?"},
    {"role": "assistant", "content": "Paris"},
    {"role": "user",      "content": "Who directed Schindler's List?"},
    {"role": "assistant", "content": "Steven Spielberg"},
]

# Few-shot for oracle_kg: shows how to read a mini-KG of triples and
# select the answer from the given facts. Covers two representative scenarios:
#   1) chain traversal (seed -> intermediate -> answer)
#   2) 1-hop candidate disambiguation (seed properties -> answer is a seed)
_FEW_SHOT_ORACLE_KG = [
    {
        "role": "user",
        "content": (
            "Use the following knowledge graph facts to answer the question.\n\n"
            "KG facts:\n"
            "(Inception) --[director]--> (Christopher Nolan)\n"
            "(Christopher Nolan) --[place of birth]--> (London)\n\n"
            "Question: What is the hometown of the director of Inception?"
        ),
    },
    {"role": "assistant", "content": "London"},
    {
        "role": "user",
        "content": (
            "Use the following knowledge graph facts to answer the question.\n\n"
            "KG facts:\n"
            "Entity: The Dark Knight\n"
            "  --[instance of]--> film\n"
            "  --[director]--> Christopher Nolan\n"
            "Entity: Heath Ledger\n"
            "  --[instance of]--> human\n"
            "  --[award received]--> Academy Award for Best Supporting Actor\n\n"
            "Question: Which actor from The Dark Knight won an Academy Award "
            "for Best Supporting Actor?"
        ),
    },
    {"role": "assistant", "content": "Heath Ledger"},
]


def build_prompt_closed_book(record: dict) -> list[dict]:
    return [
        {"role": "system",    "content": _SYSTEM_PROMPT},
        *_FEW_SHOT_CLOSED_BOOK,
        {"role": "user",      "content": record["question"]},
    ]


def build_prompt_oracle_kg(record: dict, kg_context: str) -> list[dict]:
    if kg_context:
        user_content = (
            f"Use the following knowledge graph facts to answer the question.\n\n"
            f"KG facts:\n{kg_context}\n\n"
            f"Question: {record['question']}"
        )
        few_shot = _FEW_SHOT_ORACLE_KG
    else:
        # No KG context available (no_path): falls back to closed-book
        user_content = record["question"]
        few_shot = _FEW_SHOT_CLOSED_BOOK
    return [
        {"role": "system",    "content": _SYSTEM_PROMPT},
        *few_shot,
        {"role": "user",      "content": user_content},
    ]


# ---------------------------------------------------------------------------
# LLM query
# ---------------------------------------------------------------------------

def query_llm(messages: list[dict], ollama_url: str, model: str) -> str:
    """
    Query the model via Ollama and return the cleaned response.

    Notes on reasoning-mode models (Qwen3 and similar):
    Qwen3 emits a "thinking" block by default before the content.
    On recent Ollama versions this ends up in a separate field
    ``message.thinking`` (no longer inside ``<think>...</think>`` within the
    content), which with low ``num_predict`` can consume the whole token
    budget and leave ``content`` empty. We disable thinking structurally
    via ``"think": false`` in the request body: for models without a
    thinking-mode (Gemma, Llama, Phi3) the flag is harmless and ignored.
    """
    try:
        resp = requests.post(
            f"{ollama_url}/api/chat",
            json={
                "model":    model,
                "messages": messages,
                "stream":   False,
                "think":    False,
                "options":  {"temperature": 0.0, "num_predict": 64},
            },
            timeout=240,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
    except Exception as exc:
        log.warning(f"LLM error: {exc}")
        return ""


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def _load_existing(output_path: Path) -> dict[str, dict]:
    """Load predictions already written; returns id -> record."""
    existing = {}
    if not output_path.exists():
        return existing
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            existing[r["id"]] = r
    return existing


def run(
    mode: str,
    split_path: Path,
    output_path: Path,
    subgraphs_dir: Path,
    ollama_url: str,
    model: str,
    fallback_subgraphs_dir: Path | None = None,
) -> None:
    records = []
    with open(split_path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    existing = _load_existing(output_path)
    # Consider as completed only records with non-empty pred_answer
    completed = {rid for rid, r in existing.items() if r.get("pred_answer", "")}
    todo = [r for r in records if r["id"] not in completed]

    log.info(f"Mode: {mode} | Split: {split_path} | Model: {model}")
    log.info(f"Total: {len(records)} | Already completed: {len(completed)} | To process: {len(todo)}")

    if not todo:
        log.info("No questions to process.")
        return

    stats = {"done": 0, "errors": 0}

    # Append if some records are already completed, otherwise write from scratch
    open_mode = "a" if completed else "w"
    with open(output_path, open_mode, encoding="utf-8") as out_f:
        for i, record in enumerate(todo, 1):
            if mode == "closed_book":
                messages = build_prompt_closed_book(record)
            else:
                subgraph = _load_subgraph(subgraphs_dir, record["id"],
                                        fallback_dir=fallback_subgraphs_dir)
                kg_context = build_kg_context(record, subgraph)
                messages = build_prompt_oracle_kg(record, kg_context)

            pred_answer = query_llm(messages, ollama_url, model)

            if not pred_answer:
                stats["errors"] += 1

            out_f.write(json.dumps({
                "id":             record["id"],
                "pred_answer":    pred_answer,
                "pred_qid":       None,
                "mode":           mode,
                "difficulty":     record.get("difficulty"),
                "reasoning_type": record.get("reasoning_type"),
            }, ensure_ascii=False) + "\n")

            stats["done"] += 1
            if i % 100 == 0:
                log.info(f"  {i}/{len(todo)} | errors: {stats['errors']}")

    log.info(f"Done: {stats['done']} predictions -> {output_path}")
    if stats["errors"]:
        log.warning(f"  {stats['errors']} empty predictions (LLM errors)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM baseline (via Ollama) for MetaHotpotQA"
    )
    parser.add_argument(
        "--mode", required=True, choices=["closed_book", "oracle_kg"],
        help="closed_book: question only | oracle_kg: question + gold KG context"
    )
    parser.add_argument(
        "--split", default=DEFAULT_SPLIT,
        help=f"JSONL file of the split to evaluate. Default: {DEFAULT_SPLIT}"
    )
    parser.add_argument(
        "--output", required=True,
        help="JSONL output file with predictions"
    )
    parser.add_argument(
        "--subgraphs", default=DEFAULT_SUBGRAPHS,
        help=f"Subgraphs directory (used only for oracle_kg). Default: {DEFAULT_SUBGRAPHS}"
    )
    parser.add_argument(
        "--subgraphs-fallback", default=None,
        help="Fallback subgraphs directory (searched if not found in --subgraphs)"
    )
    parser.add_argument(
        "--ollama-url", default=DEFAULT_OLLAMA_URL,
        help=f"Ollama URL. Default: {DEFAULT_OLLAMA_URL}"
    )
    parser.add_argument(
        "--ollama-model", default=DEFAULT_OLLAMA_MODEL,
        help=f"Ollama model. Default: {DEFAULT_OLLAMA_MODEL}"
    )
    args = parser.parse_args()

    if args.mode == "oracle_kg" and not Path(args.subgraphs).exists():
        log.error(f"Subgraphs directory not found: {args.subgraphs}")
        sys.exit(1)

    run(
        mode                  = args.mode,
        split_path            = Path(args.split),
        output_path           = Path(args.output),
        subgraphs_dir         = Path(args.subgraphs),
        ollama_url            = args.ollama_url,
        model                 = args.ollama_model,
        fallback_subgraphs_dir= Path(args.subgraphs_fallback) if args.subgraphs_fallback else None,
    )


if __name__ == "__main__":
    main()
