"""
fetch_wikidata_labels.py — Downloads the English Wikidata labels for every
PID, class QID, and entity QID appearing in the subgraphs under
`subgraph_dataset/` but for which we don't already have a label (ontologies
+ subgraph node labels).

Three categories of fetched identifiers:

  1. ``pid_labels`` — Property IDs (P-codes) used as edge types.
     Excludes those already in the 10 curated ontologies.

  2. ``class_labels`` — Class QIDs that appear as targets of ``TYPE`` edges.
     Excludes those already in the ontologies.

  3. ``entity_labels`` — QIDs of entities that appear as source or target of
     regular edges but lack a label in the corresponding subgraph nodes.
     They are typically nodes reached via 1-hop expansion but not downloaded
     with a full label at subgraph construction time (this happens for
     entities reached as "extra_props target" in wikidata_graphbuilder).

Output:
    utils/wikidata_labels_cache.json
    {
      "pid_labels":    {"P21": "sex or gender", ...},
      "class_labels":  {"Q4830453": "business", ...},
      "entity_labels": {"Q97025331": "...", ...},
      "fetched_at":    "2026-04-25T..."
    }

Strategy:
  - Resume: if the cache exists, skip the entities already fetched.
  - SPARQL batch (default 50 per query) + REST fallback for the unresolved.
  - Incremental cache save (crash-resilient).

Usage:
    python -m utils.fetch_wikidata_labels
    python -m utils.fetch_wikidata_labels --batch-size 30
    python -m utils.fetch_wikidata_labels --dry-run    # only count what would be needed
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from utils.ontology_utils import load_ontology_labels

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUBGRAPHS_DIR = PROJECT_ROOT / "subgraph_dataset"
CACHE_PATH = Path(__file__).resolve().parent / "wikidata_labels_cache.json"

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
REST_ENTITY_URL = "https://www.wikidata.org/wiki/Special:EntityData/{eid}.json"
HEADERS = {"User-Agent": "MetaHotpotQA-labels-fetch/1.0 (fla.valerio.96@gmail.com)"}


# ---------------------------------------------------------------------------
# Discovery: what needs to be fetched
# ---------------------------------------------------------------------------

def collect_unknown_entities() -> tuple[set[str], set[str], set[str]]:
    """
    Return (pids_to_fetch, class_qids_to_fetch, entity_qids_to_fetch).

    - PID: all P-codes appearing as edge types in the subgraphs, excluding
      those in the ontologies.
    - class QID: targets of ``TYPE`` edges, excluding those in the ontologies.
    - entity QID: source/target of regular edges for which the *containing*
      subgraph has no label in its nodes. For consistency, a QID that is
      labeled in some subgraph but not others is considered "covered" and
      is not added to the fetch (one label source is enough).
    """
    ont_labels = load_ontology_labels()
    ont_pids = set(ont_labels["pid_labels"].keys())
    ont_qids = set(ont_labels["class_labels"].keys())

    all_pids: set[str] = set()
    type_targets: set[str] = set()
    edge_qids: set[str] = set()           # all QIDs appearing in regular edges
    labelled_qids: set[str] = set()       # QIDs with a label in at least one node

    for f in SUBGRAPHS_DIR.glob("**/*.json"):
        sg = json.load(open(f, encoding="utf-8"))
        for n in sg.get("nodes", []):
            qid = n.get("qid")
            label = (n.get("label") or "").strip()
            if qid and label:
                labelled_qids.add(qid)
        for e in sg.get("edges", []):
            t = e.get("type", "")
            if t == "TYPE":
                type_targets.add(e["target"])
            else:
                all_pids.add(t)
                edge_qids.add(e["source"])
                edge_qids.add(e["target"])

    pids_unknown = all_pids - ont_pids
    classes_unknown = type_targets - ont_qids
    # Entities not labeled by any local source.
    entities_unknown = edge_qids - labelled_qids - ont_qids
    return pids_unknown, classes_unknown, entities_unknown


def load_cache() -> dict:
    if not CACHE_PATH.exists():
        return {"pid_labels": {}, "class_labels": {}, "entity_labels": {},
                "fetched_at": None}
    cache = json.load(open(CACHE_PATH, encoding="utf-8"))
    cache.setdefault("entity_labels", {})  # backward-compat with old caches
    return cache


def save_cache(cache: dict) -> None:
    cache["fetched_at"] = datetime.now(timezone.utc).isoformat()
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False, sort_keys=True)


# ---------------------------------------------------------------------------
# Fetch via SPARQL (preferred: batched)
# ---------------------------------------------------------------------------

def sparql_labels(entity_ids: list[str], retries: int = 3,
                  timeout: int = 90) -> dict[str, str]:
    """
    Batch query using the wikibase:label SERVICE (more efficient than
    rdfs:label + FILTER on entities with many languages).

    Gracefully handles:
      - 429 rate limit: exponential backoff
      - HTTP errors: retry
      - entities without a label: silently omitted from the result
    """
    if not entity_ids:
        return {}
    values_clause = " ".join(f"wd:{eid}" for eid in entity_ids)
    query = f"""
    SELECT ?e ?eLabel WHERE {{
      VALUES ?e {{ {values_clause} }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en,mul". }}
    }}
    """
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.post(  # POST handles longer queries better than GET
                SPARQL_ENDPOINT,
                data={"query": query, "format": "json"},
                headers=HEADERS,
                timeout=timeout,
            )
            if r.status_code == 429:
                wait = 10 * (attempt + 1)
                print(f"    [SPARQL 429] retry in {wait}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            out: dict[str, str] = {}
            for binding in r.json()["results"]["bindings"]:
                eid = binding["e"]["value"].split("/")[-1]
                # With wikibase:label, ?eLabel is the QID itself when no
                # label is available for the requested language; filter that out.
                lbl_raw = binding.get("eLabel", {}).get("value", "")
                lbl = lbl_raw.strip()
                if eid and lbl and lbl != eid:
                    out[eid] = lbl
            return out
        except Exception as exc:
            last_err = exc
            wait = 5 * (attempt + 1)
            print(f"    [SPARQL err attempt {attempt+1}: {exc}] retry in {wait}s")
            time.sleep(wait)
    print(f"    [SPARQL FAIL] after {retries} attempts: {last_err}")
    return {}


# ---------------------------------------------------------------------------
# Fetch via REST (fallback for single entities)
# ---------------------------------------------------------------------------

def rest_label(entity_id: str, timeout: int = 20) -> str | None:
    """
    Fetch the label of a single entity via Wikidata Special:EntityData.

    Handles:
      - redirect: the entity may have been merged into another one; in the
        response the key is the new QID, not the requested one.
      - missing en label: tries in order ``en``, ``mul``, then any
        available language (the baseline runs a multilingual LLM, so any
        label is better than a raw QID).
    """
    try:
        r = requests.get(REST_ENTITY_URL.format(eid=entity_id),
                         headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        entities = data.get("entities", {})
        if not entities:
            print(f"    [REST {entity_id} no entities]")
            return None
        # Direct lookup; if missing, take the first entry (redirect case)
        ent = entities.get(entity_id) or next(iter(entities.values()))
        labels = ent.get("labels", {})
        if not labels:
            print(f"    [REST {entity_id} no labels at all]")
            return None
        # Priority: en, mul, any
        for lang in ("en", "mul"):
            if lang in labels:
                val = labels[lang].get("value", "").strip()
                if val:
                    return val
        # Fallback: first available language (with language prefix for clarity)
        first_lang, first_label = next(iter(labels.items()))
        val = (first_label.get("value") or "").strip()
        if val:
            return f"{val} [{first_lang}]"
        return None
    except Exception as exc:
        print(f"    [REST {entity_id} fail] {exc}")
        return None


# ---------------------------------------------------------------------------
# Main fetch loop
# ---------------------------------------------------------------------------

def fetch_in_batches(entity_ids: list[str], cache_key: str, cache: dict,
                     batch_size: int = 50, sleep_between: float = 1.0) -> None:
    """Fetch the labels and save incrementally into cache[cache_key]."""
    if not entity_ids:
        return
    total = len(entity_ids)
    print(f"\nFetching {total} {cache_key} in batches of {batch_size}...")
    for i in range(0, total, batch_size):
        chunk = entity_ids[i:i + batch_size]
        print(f"  batch {i//batch_size + 1}/{(total + batch_size - 1)//batch_size}: "
              f"{len(chunk)} entities (range {chunk[0]}..{chunk[-1]})")

        labels = sparql_labels(chunk)
        # REST fallback for the entities SPARQL did not resolve
        missing_after_sparql = [e for e in chunk if e not in labels]
        if missing_after_sparql:
            print(f"    SPARQL resolved {len(labels)}/{len(chunk)}, "
                  f"REST fallback on {len(missing_after_sparql)}")
            for e in missing_after_sparql:
                lbl = rest_label(e)
                if lbl:
                    labels[e] = lbl
                time.sleep(0.3)

        cache[cache_key].update(labels)
        save_cache(cache)
        print(f"    added {len(labels)} labels, total cache[{cache_key}]={len(cache[cache_key])}")
        time.sleep(sleep_between)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"Cache path: {CACHE_PATH}")
    print(f"Subgraphs:  {SUBGRAPHS_DIR}\n")

    pids_unknown, classes_unknown, entities_unknown = collect_unknown_entities()
    cache = load_cache()
    cached_pids = set(cache["pid_labels"].keys())
    cached_classes = set(cache["class_labels"].keys())
    cached_entities = set(cache["entity_labels"].keys())

    pids_todo = sorted(pids_unknown - cached_pids)
    classes_todo = sorted(classes_unknown - cached_classes)
    entities_todo = sorted(entities_unknown - cached_entities)

    print(f"To discover (subgraphs - local sources):")
    print(f"  PID:       {len(pids_unknown)}")
    print(f"  class QID: {len(classes_unknown)}")
    print(f"  entity QID:{len(entities_unknown)}")
    print(f"Already in cache:")
    print(f"  PID:       {len(cached_pids & pids_unknown)}")
    print(f"  class QID: {len(cached_classes & classes_unknown)}")
    print(f"  entity QID:{len(cached_entities & entities_unknown)}")
    print(f"To fetch now:")
    print(f"  PID:       {len(pids_todo)}")
    print(f"  class QID: {len(classes_todo)}")
    print(f"  entity QID:{len(entities_todo)}")

    if args.dry_run:
        print(f"\n(dry-run)")
        return

    fetch_in_batches(pids_todo, "pid_labels", cache, batch_size=args.batch_size)
    fetch_in_batches(classes_todo, "class_labels", cache, batch_size=args.batch_size)
    fetch_in_batches(entities_todo, "entity_labels", cache, batch_size=args.batch_size)

    print(f"\nDone.")
    print(f"  pid_labels    in cache: {len(cache['pid_labels'])}")
    print(f"  class_labels  in cache: {len(cache['class_labels'])}")
    print(f"  entity_labels in cache: {len(cache['entity_labels'])}")
    print(f"  saved to {CACHE_PATH}")


if __name__ == "__main__":
    main()
