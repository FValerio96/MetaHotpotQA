"""
Answer Finder: locates the answer node in the subgraph for each MetaHotpotQA
question.

For each question:
1. Load the corresponding subgraph
2. Search the answer in node labels/descriptions across 4 tiers:
   - Tier 1: case-insensitive exact match on label or description
   - Tier 2: normalized match (no punctuation, lowercase)
   - Tier 3: substring or token overlap > threshold -> LLM disambiguation
   - Tier 4: no match -> not found
3. Output: found.jsonl and not_found.jsonl

Usage:
    python 01_find_answers.py [--dataset PATH] [--subgraphs DIR] [--output DIR]
                              [--threshold 0.5] [--ollama-url URL] [--ollama-model NAME]
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

import requests


# ============================================================================
# NORMALIZATION AND MATCHING
# ============================================================================

def normalize(text: str) -> str:
    """Lowercase, remove punctuation, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def token_overlap(a: str, b: str) -> float:
    """Jaccard similarity over tokens (after normalization)."""
    tokens_a = set(a.split())
    tokens_b = set(b.split())
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def tier3_score(norm_answer: str, norm_label: str, threshold: float) -> float:
    """
    Return a score [0,1] if we're in Tier 3, otherwise 0.
    Used to pick the best candidate among multiple tier-3 nodes.
    """
    # substring (with min-length guard to avoid false positives)
    if len(norm_answer) >= 3 and len(norm_label) >= 3:
        if norm_answer in norm_label:
            return 0.5 + len(norm_answer) / (len(norm_label) + 1)
        if norm_label in norm_answer:
            return 0.5 + len(norm_label) / (len(norm_answer) + 1)
    # token overlap
    overlap = token_overlap(norm_answer, norm_label)
    if overlap >= threshold:
        return overlap
    return 0.0


def match_node(answer: str, node: dict, threshold: float) -> tuple[int, float]:
    """
    Check the node's label and description.
    Return (best_tier, score_tier3) for this node.
    score_tier3 is relevant only if best_tier == 3.
    """
    answer_lower = answer.lower().strip()
    norm_answer = normalize(answer)
    best_tier = 4
    best_score = 0.0

    for text in [node.get("label") or "", node.get("description") or ""]:
        if not text:
            continue

        # Tier 1
        if answer_lower == text.lower().strip():
            return 1, 1.0

        # Tier 2
        if norm_answer == normalize(text):
            best_tier = min(best_tier, 2)
            best_score = 1.0
            continue

        # Tier 3
        score = tier3_score(norm_answer, normalize(text), threshold)
        if score > 0.0 and best_tier > 2:
            best_tier = 3
            best_score = max(best_score, score)

    return best_tier, best_score


def find_answer_in_subgraph(
    answer: str, nodes: list, threshold: float
) -> tuple[Optional[str], int, list[str]]:
    """
    Search the answer in all nodes of the subgraph.
    Return (qid, tier, tier3_candidates).

    - Tier 1/2: qid is the matched node, tier3_candidates is empty.
    - Tier 3:   qid is None, tier3_candidates contains ALL candidate QIDs
                sorted by descending score (the LLM is queried on each).
    - Tier 4:   (None, 4, [])

    Strategy:
    - First pass: look for tier 1 or tier 2, return on the first hit.
    - Second pass (only if no tier 1/2): collect all tier-3 candidates and
      sort them by descending score.
    """
    # First pass: tier 1 and 2
    for node in nodes:
        tier, _ = match_node(answer, node, threshold)
        if tier <= 2:
            return node["qid"], tier, []

    # Second pass: all tier-3 candidates sorted by score
    tier3: list[tuple[float, str]] = []
    for node in nodes:
        tier, score = match_node(answer, node, threshold)
        if tier == 3:
            tier3.append((score, node["qid"]))

    if tier3:
        tier3.sort(key=lambda x: x[0], reverse=True)
        return None, 3, [qid for _, qid in tier3]

    return None, 4, []


# ============================================================================
# LLM DISAMBIGUATION (tier 3 only)
# ============================================================================

_FEW_SHOT_MESSAGES = [
    {
        "role": "user",
        "content": (
            'Do these two strings refer to the same real-world entity?\n'
            'A: "Robert Downey Junior"\n'
            'B: "Robert D. Junior"\n'
            'Answer with only "yes" or "no".'
        ),
    },
    {"role": "assistant", "content": "yes"},
    {
        "role": "user",
        "content": (
            'Do these two strings refer to the same real-world entity?\n'
            'A: "Apple"\n'
            'B: "Google"\n'
            'Answer with only "yes" or "no".'
        ),
    },
    {"role": "assistant", "content": "no"},
]


def ask_llm(
    answer: str,
    candidate_label: str,
    ollama_url: str,
    model: str,
) -> bool:
    """
    Ask Qwen3 via Ollama whether answer and candidate_label refer to the
    same entity. Return True if confirmed, False otherwise (including network errors).
    """
    prompt = (
        f'Do these two strings refer to the same real-world entity?\n'
        f'A: "{answer}"\n'
        f'B: "{candidate_label}"\n'
        f'Answer with only "yes" or "no".'
    )
    try:
        resp = requests.post(
            f"{ollama_url}/api/chat",
            json={
                "model": model,
                "messages": [
                    # /no_think disables Qwen3's reasoning chain
                    # (ignored by other models, harmless)
                    {"role": "system", "content": "/no_think"},
                    *_FEW_SHOT_MESSAGES,
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "options": {"temperature": 0.0},
            },
            timeout=60,
        )
        resp.raise_for_status()
        raw = resp.json()["message"]["content"]
        # Strip any <think>...</think> tag produced by Qwen3
        reply = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip().lower()
        if "yes" in reply:
            return True
        if "no" in reply:
            return False
        print(f"[WARN] ambiguous LLM reply: {raw!r}", file=sys.stderr)
        return False
    except Exception as exc:
        print(f"[WARN] LLM error: {exc}", file=sys.stderr)
        return False


def get_node_label(nodes: list, qid: str) -> str:
    """Retrieve the label of a node given its QID."""
    for node in nodes:
        if node["qid"] == qid:
            return node.get("label") or qid
    return qid


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process(
    dataset_path: str,
    subgraphs_dir: str,
    output_dir: str,
    threshold: float = 0.5,
    ollama_url: str = "http://127.0.0.1:11434",
    ollama_model: str = "qwen3:14b-q4_K_M",
) -> None:
    dataset_path = Path(dataset_path)
    subgraphs_dir = Path(subgraphs_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    found_path = output_dir / "found.jsonl"
    not_found_path = output_dir / "not_found.jsonl"

    stats = {
        "total": 0,
        "tier1": 0,
        "tier2": 0,
        "tier3_yes": 0,
        "tier3_no": 0,
        "tier4": 0,
        "no_subgraph": 0,
    }

    with (
        open(dataset_path) as fin,
        open(found_path, "w") as f_found,
        open(not_found_path, "w") as f_not_found,
    ):
        for line in fin:
            example = json.loads(line)
            stats["total"] += 1
            example_id = example["id"]
            answer = example["answer"]

            subgraph_path = subgraphs_dir / f"{example_id}.json"
            if not subgraph_path.exists():
                stats["no_subgraph"] += 1
                f_not_found.write(
                    json.dumps({**example, "reason": "no_subgraph"}) + "\n"
                )
                continue

            with open(subgraph_path) as sg_file:
                subgraph = json.load(sg_file)

            nodes = subgraph.get("nodes", [])
            match_qid, tier, tier3_candidates = find_answer_in_subgraph(answer, nodes, threshold)

            if tier == 1:
                stats["tier1"] += 1
                f_found.write(
                    json.dumps({**example, "answer_node_qid": match_qid, "match_tier": 1}) + "\n"
                )

            elif tier == 2:
                stats["tier2"] += 1
                f_found.write(
                    json.dumps({**example, "answer_node_qid": match_qid, "match_tier": 2}) + "\n"
                )

            elif tier == 3:
                confirmed_qid = None
                for candidate_qid in tier3_candidates:
                    candidate_label = get_node_label(nodes, candidate_qid)
                    if ask_llm(answer, candidate_label, ollama_url, ollama_model):
                        confirmed_qid = candidate_qid
                        break
                if confirmed_qid:
                    stats["tier3_yes"] += 1
                    f_found.write(
                        json.dumps({**example, "answer_node_qid": confirmed_qid, "match_tier": 3}) + "\n"
                    )
                else:
                    stats["tier3_no"] += 1
                    f_not_found.write(
                        json.dumps({**example, "reason": "tier3_llm_no"}) + "\n"
                    )

            else:
                stats["tier4"] += 1
                f_not_found.write(
                    json.dumps({**example, "reason": "no_match"}) + "\n"
                )

            if stats["total"] % 500 == 0:
                _print_progress(stats)

    _print_final(stats, found_path, not_found_path)


def process_tier3(
    not_found_path: str,
    subgraphs_dir: str,
    output_dir: str,
    threshold: float = 0.5,
    ollama_url: str = "http://127.0.0.1:11434",
    ollama_model: str = "qwen3:14b-q4_K_M",
) -> None:
    """Re-apply ask_llm on the tier3_llm_no entries of an existing not_found.jsonl."""
    not_found_path = Path(not_found_path)
    subgraphs_dir = Path(subgraphs_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    found_path = output_dir / "tier3_retry_found.jsonl"
    still_not_found_path = output_dir / "tier3_retry_not_found.jsonl"

    stats = {"total": 0, "skipped": 0, "tier3_yes": 0, "tier3_no": 0}

    with (
        open(not_found_path) as fin,
        open(found_path, "w") as f_found,
        open(still_not_found_path, "w") as f_not_found,
    ):
        for line in fin:
            example = json.loads(line)
            if example.get("reason") != "tier3_llm_no":
                stats["skipped"] += 1
                continue

            stats["total"] += 1
            example_id = example["id"]
            answer = example["answer"]

            subgraph_path = subgraphs_dir / f"{example_id}.json"
            if not subgraph_path.exists():
                f_not_found.write(json.dumps({**example, "reason": "no_subgraph"}) + "\n")
                stats["tier3_no"] += 1
                continue

            with open(subgraph_path) as sg_file:
                nodes = json.load(sg_file).get("nodes", [])

            _, tier, tier3_candidates = find_answer_in_subgraph(answer, nodes, threshold)
            if tier != 3:
                # No tier-3 candidate at this threshold — skip
                f_not_found.write(json.dumps({**example, "reason": "no_tier3_candidate"}) + "\n")
                stats["tier3_no"] += 1
                continue

            confirmed_qid = None
            for candidate_qid in tier3_candidates:
                candidate_label = get_node_label(nodes, candidate_qid)
                if ask_llm(answer, candidate_label, ollama_url, ollama_model):
                    confirmed_qid = candidate_qid
                    break
            if confirmed_qid:
                stats["tier3_yes"] += 1
                example.pop("reason", None)
                f_found.write(json.dumps({**example, "answer_node_qid": confirmed_qid, "match_tier": 3}) + "\n")
            else:
                stats["tier3_no"] += 1
                f_not_found.write(json.dumps({**example, "reason": "tier3_llm_no"}) + "\n")

            if stats["total"] % 200 == 0:
                print(f"[{stats['total']}] yes:{stats['tier3_yes']} no:{stats['tier3_no']}")

    print("\n=== TIER3 RETRY DONE ===")
    print(f"Processed   : {stats['total']}  (non-tier3 skipped: {stats['skipped']})")
    print(f"Found       : {stats['tier3_yes']}  ({100*stats['tier3_yes']/max(stats['total'],1):.1f}%)")
    print(f"Not found   : {stats['tier3_no']}")
    print(f"\nOutput: {found_path}")
    print(f"        {still_not_found_path}")


def _print_progress(stats: dict) -> None:
    t = stats["total"]
    print(
        f"[{t}] "
        f"T1:{stats['tier1']} "
        f"T2:{stats['tier2']} "
        f"T3+:{stats['tier3_yes']} "
        f"T3-:{stats['tier3_no']} "
        f"T4:{stats['tier4']} "
        f"NoSG:{stats['no_subgraph']}"
    )


def _print_final(stats: dict, found_path: Path, not_found_path: Path) -> None:
    found = stats["tier1"] + stats["tier2"] + stats["tier3_yes"]
    not_found = stats["tier3_no"] + stats["tier4"] + stats["no_subgraph"]
    print("\n=== DONE ===")
    print(f"Total questions :  {stats['total']}")
    print(f"Found           :  {found}  ({100*found/max(stats['total'],1):.1f}%)")
    print(f"  Tier 1 (exact):  {stats['tier1']}")
    print(f"  Tier 2 (norm) :  {stats['tier2']}")
    print(f"  Tier 3 (LLM+) :  {stats['tier3_yes']}")
    print(f"Not found       :  {not_found}  ({100*not_found/max(stats['total'],1):.1f}%)")
    print(f"  Tier 3 (LLM-) :  {stats['tier3_no']}")
    print(f"  Tier 4        :  {stats['tier4']}")
    print(f"  No subgraph   :  {stats['no_subgraph']}")
    print(f"\nOutput: {found_path}")
    print(f"        {not_found_path}")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search the answer node in MetaHotpotQA subgraphs.")
    parser.add_argument(
        "--dataset",
        default="data/hotpot_matched.jsonl",
        help="Path to the JSONL file of questions (output of Stage 1b)",
    )
    parser.add_argument(
        "--subgraphs",
        default="data/subgraphs",
        help="Directory containing the subgraph JSON files (output of Stage 2)",
    )
    parser.add_argument(
        "--output",
        default="data/answer_search",
        help="Output directory for found.jsonl and not_found.jsonl",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Jaccard threshold for token overlap (Tier 3, default: 0.5)",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://127.0.0.1:11434",
        help="Base URL of the Ollama server",
    )
    parser.add_argument(
        "--ollama-model",
        default="qwen3:14b-q4_K_M",
        help="Ollama model name",
    )
    parser.add_argument(
        "--retry-tier3",
        action="store_true",
        help="Re-apply ask_llm on the tier3_llm_no entries of a not_found.jsonl (pass its path via --dataset)",
    )
    args = parser.parse_args()

    if args.retry_tier3:
        process_tier3(
            not_found_path=args.dataset,
            subgraphs_dir=args.subgraphs,
            output_dir=args.output,
            threshold=args.threshold,
            ollama_url=args.ollama_url,
            ollama_model=args.ollama_model,
        )
    else:
        process(
            dataset_path=args.dataset,
            subgraphs_dir=args.subgraphs,
            output_dir=args.output,
            threshold=args.threshold,
            ollama_url=args.ollama_url,
            ollama_model=args.ollama_model,
        )
