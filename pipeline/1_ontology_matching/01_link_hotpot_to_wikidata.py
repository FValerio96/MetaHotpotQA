"""
Stage 1a — Link HotpotQA supporting facts to Wikidata QIDs and P31 classes.

Pipeline:
  1. Download HotpotQA train + dev splits (cached locally).
  2. Normalize supporting_facts into (title, sentence_idx) entity stubs.
  3. First pass (parallel): query Wikidata wbgetentities by enwiki site title
     to obtain QID + P31 classes for each entity.
  4. Fixup pass (parallel): for entities that did not resolve, query the
     Wikipedia API with redirect resolution to recover the QID, then fetch
     classes.

Output JSONL: one record per HotpotQA question, augmented with an `entities`
list of {title, sentence_idx, qid, classes} dicts. Records preserve all
original HotpotQA fields (id, question, answer, type, level, context,
supporting_facts).

Usage:
  python 01_link_hotpot_to_wikidata.py --out data/hotpot_linked.jsonl
"""

import argparse
import json
import os
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

HOTPOT_URLS = {
    "train": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json",
    "dev":   "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
}


def _user_agent() -> str:
    contact = os.environ.get("WIKIDATA_CONTACT", "").strip()
    if not contact:
        raise RuntimeError(
            "WIKIDATA_CONTACT env var is required (set it to a contact email "
            "or URL identifying you, per Wikidata's User-Agent policy).")
    return f"MetaHotpotQA-linker/1.0 (https://github.com/FValerio96/MetaHotpot; mailto:{contact})"


HEADERS = {"User-Agent": _user_agent()}
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"

_title_cache: dict[str, dict | None] = {}
_classes_cache: dict[str, list[str]] = {}
_cache_lock = threading.Lock()


def download_hotpot(cache_dir: Path) -> dict[str, list]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for split, url in HOTPOT_URLS.items():
        path = cache_dir / f"hotpot_{split}.json"
        if not path.exists():
            print(f"  downloading {url}")
            urllib.request.urlretrieve(url, path)
        with open(path) as f:
            out[split] = json.load(f)
        print(f"  {split}: {len(out[split])} questions")
    return out


def normalize_record(item: dict) -> dict:
    """Flatten supporting_facts into a deduplicated entity list with sentence indices.
    Also renames HotpotQA's `type` field (bridge|comparison) to `reasoning_type`,
    which is the field name used downstream by the path annotator."""
    seen: dict[str, list[int]] = {}
    for title, sent_idx in item.get("supporting_facts", []):
        seen.setdefault(title, []).append(sent_idx)
    entities = [{"title": t, "sentence_idx": idx} for t, idxs in seen.items() for idx in idxs]
    return {
        **item,
        "entities": entities,
        "id": item["_id"],
        "reasoning_type": item["type"],
    }


def fetch_by_enwiki_title(title: str, retries: int = 3) -> dict | None:
    """Wikidata wbgetentities by enwiki site title → {qid, classes} or None."""
    with _cache_lock:
        if title in _title_cache:
            return _title_cache[title]

    params = {
        "action": "wbgetentities",
        "sites": "enwiki",
        "titles": title,
        "props": "claims",
        "format": "json",
        "maxlag": 5,
    }
    for attempt in range(retries):
        try:
            r = requests.get(WIKIDATA_API, params=params, headers=HEADERS, timeout=10)
            data = r.json()
            if isinstance(data, dict) and "error" in data and "maxlag" in data["error"].get("code", ""):
                time.sleep(5 * (attempt + 1))
                continue
            if r.status_code != 200:
                break

            entities = data.get("entities", {})
            if not entities:
                break
            qid, entity = next(iter(entities.items()))
            if qid.startswith("-1"):
                break

            classes = []
            for claim in entity.get("claims", {}).get("P31", []):
                try:
                    classes.append(claim["mainsnak"]["datavalue"]["value"]["id"])
                except KeyError:
                    continue
            result = {"qid": qid, "classes": classes}
            with _cache_lock:
                _title_cache[title] = result
            return result
        except (requests.RequestException, ValueError):
            time.sleep(2 * (attempt + 1))

    with _cache_lock:
        _title_cache[title] = None
    return None


def qid_from_enwiki_redirect(title: str, retries: int = 3) -> tuple[str | None, str | None]:
    """Wikipedia API with redirects → (qid, canonical_title)."""
    for attempt in range(retries):
        try:
            r = requests.get(WIKIPEDIA_API, params={
                "action": "query", "format": "json", "titles": title,
                "redirects": 1, "prop": "pageprops",
            }, headers=HEADERS, timeout=10)
            if r.status_code != 200:
                time.sleep(2 * (attempt + 1))
                continue
            pages = r.json().get("query", {}).get("pages", {})
            page = next(iter(pages.values()), None)
            if not page or "missing" in page:
                return None, None
            return page.get("pageprops", {}).get("wikibase_item"), page.get("title")
        except (requests.RequestException, ValueError):
            time.sleep(2 * (attempt + 1))
    return None, None


def classes_from_qid(qid: str, retries: int = 3) -> list[str]:
    with _cache_lock:
        if qid in _classes_cache:
            return _classes_cache[qid]
    for attempt in range(retries):
        try:
            r = requests.get(WIKIDATA_API, params={
                "action": "wbgetentities", "ids": qid,
                "props": "claims", "format": "json",
            }, headers=HEADERS, timeout=10)
            if r.status_code != 200:
                time.sleep(2 * (attempt + 1))
                continue
            entity = r.json().get("entities", {}).get(qid)
            if not entity:
                return []
            classes = []
            for claim in entity.get("claims", {}).get("P31", []):
                try:
                    classes.append(claim["mainsnak"]["datavalue"]["value"]["id"])
                except KeyError:
                    continue
            with _cache_lock:
                _classes_cache[qid] = classes
            return classes
        except (requests.RequestException, ValueError):
            time.sleep(2 * (attempt + 1))
    return []


def enrich_entity_first_pass(entity: dict) -> dict:
    info = fetch_by_enwiki_title(entity["title"])
    if info is None:
        return {**entity, "qid": None, "classes": []}
    return {**entity, "qid": info["qid"], "classes": info["classes"]}


def enrich_entity_fixup(entity: dict) -> dict:
    """Recover qid via Wikipedia redirect, then fetch classes."""
    if entity.get("qid"):
        return entity
    qid, canonical = qid_from_enwiki_redirect(entity["title"])
    if not qid:
        return entity
    out = {**entity, "qid": qid}
    if canonical:
        out["title"] = canonical.strip()
    out["classes"] = classes_from_qid(qid)
    return out


def process_records(records: list[dict], out_path: Path, max_workers: int = 20) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fout, ThreadPoolExecutor(max_workers=max_workers) as ex:
        for line_num, item in enumerate(records, start=1):
            rec = normalize_record(item)
            rec["entities"] = list(ex.map(enrich_entity_first_pass, rec["entities"]))
            rec["entities"] = list(ex.map(enrich_entity_fixup, rec["entities"]))
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if line_num % 100 == 0:
                print(f"  processed {line_num}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", default="data/_hotpot_cache",
                   help="HotpotQA download cache directory.")
    p.add_argument("--out", default="data/hotpot_linked.jsonl",
                   help="Output JSONL path.")
    p.add_argument("--workers", type=int, default=20,
                   help="Thread pool size for Wikidata/Wikipedia API calls.")
    args = p.parse_args()

    splits = download_hotpot(Path(args.cache_dir))
    all_records = splits["train"] + splits["dev"]
    print(f"Total HotpotQA records: {len(all_records)}")
    process_records(all_records, Path(args.out), max_workers=args.workers)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
