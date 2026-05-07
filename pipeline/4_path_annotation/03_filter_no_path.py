"""
filter_no_path.py — Rimuove le domande no_path dagli split MetaHotpotQA.

Scelta di progettazione: le domande con difficulty="no_path" hanno l'entità
risposta presente nel subgrafo (copertura garantita) ma non esiste alcun path
strutturale che connetta le seed entity alla risposta entro i limiti di hop
configurati. Poiché nessun sistema KG-based può rispondere a queste domande
navigando il grafo, e poiché l'oracle KG context serializzato è vuoto per questi
record (nessun path da serializzare), mantenerle nel dataset introdurrebbe esempi
strutturalmente irrisolvibili. Vengono quindi rimosse come scelta esplicita di
progettazione dopo la fase di path annotation.

Uso:
    python filter_no_path.py [--splits DIR] [--output DIR]

Se --output è omesso, le split vengono riscritte in-place.
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[filter_no_path] %(message)s")
log = logging.getLogger(__name__)

SPLIT_FILES = ("train.jsonl", "dev.jsonl", "test.jsonl")


def filter_split(src: Path, dst: Path) -> tuple[int, int]:
    """Filtra un file JSONL rimuovendo i record no_path. Ritorna (totale, rimossi)."""
    kept, removed = [], 0
    with open(src, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("difficulty") == "no_path":
                removed += 1
            else:
                kept.append(r)
    with open(dst, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(kept) + removed, removed


def update_summary(splits_dir: Path) -> None:
    """Ricalcola summary.json dai file filtrati."""
    summary_path = splits_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
    else:
        summary = {}

    counts: dict[str, int] = {}
    diff_dist: dict[str, int] = defaultdict(int)
    total = 0

    for fname in SPLIT_FILES:
        path = splits_dir / fname
        if not path.exists():
            continue
        split_name = fname.replace(".jsonl", "")
        n = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                n += 1
                diff_dist[r.get("difficulty", "unknown")] += 1
        counts[split_name] = n
        total += n

    summary.update({
        "total": total,
        "no_path_removed": True,
        **counts,
        "difficulty_distribution": dict(diff_dist),
    })
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log.info(f"Summary aggiornato → {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rimuove le domande no_path dagli split MetaHotpotQA"
    )
    parser.add_argument(
        "--splits",
        default="data/splits_patched",
        help="Directory degli split. Default: %(default)s",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Directory di output (default: sovrascrive --splits in-place)",
    )
    args = parser.parse_args()

    src_dir = Path(args.splits)
    dst_dir = Path(args.output) if args.output else src_dir
    dst_dir.mkdir(parents=True, exist_ok=True)

    total_removed = 0
    for fname in SPLIT_FILES:
        src = src_dir / fname
        if not src.exists():
            log.warning(f"{fname} non trovato, saltato.")
            continue
        dst = dst_dir / fname
        total, removed = filter_split(src, dst)
        log.info(f"  {fname}: {total} totali → {removed} no_path rimossi → {total - removed} rimasti")
        total_removed += removed

    log.info(f"Totale rimossi: {total_removed}")
    update_summary(dst_dir)


if __name__ == "__main__":
    main()
