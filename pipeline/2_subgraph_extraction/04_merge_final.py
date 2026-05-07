"""
build_subgraphs_final.py — Merge `subgraphs_extended/` + `subgraphs_s0_patched/`
in `subgraphs_final/`, la directory canonica dei subgrafi usata dagli esperimenti.

Logica:
  - base: tutti i file JSON da subgraphs_extended/ vengono copiati in subgraphs_final/
  - override: i file in subgraphs_s0_patched/ (subgrafi arricchiti dal bridge
    patcher con edge SPARQL aggiuntivi) sovrascrivono quelli omonimi della base.

Il contenuto dei subgrafi patched è un superset di quello extended (stessi nodi
base + edge SPARQL aggiunti, eventualmente con nodi intermedi 2-hop), quindi
l'overwrite è safety-preserving.

Uso:
    python pipeline/2_subgraph_extraction/04_merge_final.py
    python pipeline/2_subgraph_extraction/04_merge_final.py --extended data/subgraphs \\
                                                             --patched data/subgraphs_patched \\
                                                             --output data/subgraphs_final
"""

import argparse
import shutil
from pathlib import Path


def build_final(extended_dir: Path, patched_dir: Path, output_dir: Path,
                force: bool = False) -> dict:
    if output_dir.exists() and not force:
        raise SystemExit(
            f"{output_dir} esiste già. Rilancia con --force per sovrascrivere."
        )
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    stats = {"base_copied": 0, "overridden": 0, "patched_not_in_base": 0}

    # 1. Base: copia subgraphs_extended
    for src in extended_dir.glob("*.json"):
        shutil.copy2(src, output_dir / src.name)
        stats["base_copied"] += 1

    # 2. Override: sovrascrivi con i subgraphs_s0_patched
    base_names = {p.name for p in output_dir.glob("*.json")}
    for src in patched_dir.glob("*.json"):
        if src.name in base_names:
            stats["overridden"] += 1
        else:
            stats["patched_not_in_base"] += 1
        shutil.copy2(src, output_dir / src.name)

    stats["total_final"] = sum(1 for _ in output_dir.glob("*.json"))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Costruisce subgraphs_final/ come merge extended + patched"
    )
    parser.add_argument("--extended", default="data/subgraphs",
                        help="Directory dei subgrafi base ontology-driven (output di 02_build_subgraphs)")
    parser.add_argument("--patched", default="data/subgraphs_patched",
                        help="Directory dei subgrafi arricchiti dal bridge patcher (output di 4/02_bridge_patcher)")
    parser.add_argument("--output", default="data/subgraphs_final",
                        help="Directory di output, archiviata su Zenodo")
    parser.add_argument("--force", action="store_true",
                        help="Sovrascrive la directory di output se esiste")
    args = parser.parse_args()

    extended = Path(args.extended)
    patched = Path(args.patched)
    output = Path(args.output)

    if not extended.exists():
        raise SystemExit(f"{extended} non esiste")
    if not patched.exists():
        raise SystemExit(f"{patched} non esiste")

    print(f"Base:     {extended}")
    print(f"Override: {patched}")
    print(f"Output:   {output}\n")

    stats = build_final(extended, patched, output, force=args.force)

    print(f"Subgrafi base copiati:              {stats['base_copied']}")
    print(f"Subgrafi sovrascritti da patched:   {stats['overridden']}")
    print(f"Patched non presenti nella base:    {stats['patched_not_in_base']}")
    print(f"Totale in {output}: {stats['total_final']}")


if __name__ == "__main__":
    main()
