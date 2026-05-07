# MetaHotpotQA

> A schema-aware multi-hop Knowledge Graph Question Answering (KGQA) benchmark — 12,056 human-authored questions paired with per-question Wikidata subgraphs, ontology-typed entities, and recovered reasoning paths.

[![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)
[![Code: MIT](https://img.shields.io/badge/Code-MIT-blue.svg)](LICENSE)
[![Dataset: Zenodo](https://img.shields.io/badge/Dataset-Zenodo-3a4b9e.svg)](https://doi.org/10.5281/zenodo.20071935)
[![Dataset: HuggingFace](https://img.shields.io/badge/Dataset-HuggingFace-yellow.svg)](TODO_HF_LINK)

---

## Resources

| Resource | Link |
|---|---|
| **Paper** (ISWC 2026 Resources Track) | TODO_PAPER_LINK |
| **Dataset (canonical, with subgraphs)** | [10.5281/zenodo.20071935](https://doi.org/10.5281/zenodo.20071935) |
| **Dataset (splits only, easy load)** | TODO_HF_LINK |
| **Code & pipeline** (this repo) | this repo |

The dataset itself lives on **Zenodo** (canonical citation, includes the per-question Wikidata subgraphs, ~3 GB). The same splits are mirrored on **HuggingFace Hub** for use with `datasets.load_dataset`. This repository hosts the **construction pipeline** and the **evaluation code**.

---

## Highlights

- **12,056 questions** sourced from HotpotQA, grounded into Wikidata
- **10 ontological domains** (Movie, Music, Book, Sport, Military, Computer, Space, Politics, Nature, Culture) from Text2KGBench
- **3 difficulty categories** with verified reasoning paths: `entity_selection`, `traversal`, `property_comparison`
- **100% answer coverage** in the released subgraphs, by construction
- **Stratified splits** (train 8,439 / dev 1,809 / test 1,808), seed `42`
- **Path-annotated**: explicit `supporting_paths` for traversal and entity-selection records (4-hop max, ontology-filtered)
- **Validated** with four 4-bit quantized LLM baselines (Qwen3-14B, Gemma3-12B, Llama3.1-8B, Phi3-14B): mean +48.5 pp EM gain over closed-book

---

## Quick Start

### Install

```bash
git clone https://github.com/<org>/MetaHotpotQA.git
cd MetaHotpotQA
pip install -r requirements.txt
```

### Get the data

**Option A — HuggingFace (splits only, ~7 MB):**

```python
from datasets import load_dataset
ds = load_dataset("TODO_HF_HANDLE/MetaHotpotQA")
```

**Option B — Zenodo (canonical, ~550 MB with subgraphs):**

```bash
mkdir -p data && cd data
wget https://zenodo.org/records/20071935/files/published_splits.zip
wget https://zenodo.org/records/20071935/files/subgraphs_dataset.tar.gz
unzip published_splits.zip -d published_splits/
tar -xzf subgraphs_dataset.tar.gz
```

After extraction you should have:
- `data/published_splits/{train,dev,test}.jsonl` — the dataset records
- `data/subgraph_dataset/{train,dev,test}/<id>.json` — per-question Wikidata subgraphs

The 10 ontologies and the construction pipeline live in this repository under [ontologies/](ontologies/) and [pipeline/](pipeline/).

### Load and inspect

```python
import json

def load(p):
    with open(p) as f:
        return [json.loads(line) for line in f]

test = load("data/published_splits/test.jsonl")
print(test[0])
# {'id': '...', 'question': '...', 'answer': '...', 'answer_node_qid': 'Q...',
#  'entity_qids': ['Q...', 'Q...'], 'reasoning_type': 'bridge',
#  'difficulty': 'traversal', 'strategy_used': 1, 'match_tier': 1,
#  'matched_ontologies': [{'ont_id': 'ont_4_book', 'matched_qids': ['Q...']}],
#  'supporting_paths': [{'length': 1, 'triples': [['Q...', 'P36', 'Q...']]}]}
```

See [docs/DATASET_CARD.md](docs/DATASET_CARD.md) for the full schema specification.

---

## Reproducing the Paper Results

The four baselines reported in the paper (Tables 1–4) can be re-evaluated against the released predictions in `baselines/results/`. To re-run inference yourself, install [Ollama](https://ollama.ai), pull the four models, and run:

```bash
# Closed-book
python -m baselines.llm_baseline --mode closed_book \
    --ollama-model qwen3:14b \
    --split data/published_splits/test.jsonl \
    --output preds_qwen3_14b_cb.jsonl

# Oracle KG (uses supporting_paths + 1-hop neighborhood from subgraphs)
python -m baselines.llm_baseline --mode oracle_kg \
    --ollama-model qwen3:14b \
    --split data/published_splits/test.jsonl \
    --subgraphs data/subgraph_dataset/test/ \
    --output preds_qwen3_14b_oracle.jsonl
```

To compute metrics (EM, F1) on a prediction file (Tables 1, 3, 4):

```bash
python -m baselines.evaluate \
    --predictions baselines/results/oracle/preds_qwen3_14b.jsonl \
    --split data/published_splits/test.jsonl
```

For the `oracle_full` vs `oracle_partial` breakdown (Table 2), point at the directory containing the per-model `preds_*.jsonl` files:

```bash
python -m baselines.evaluate_oracle_breakdown \
    --results_dir baselines/results/oracle/ \
    --split data/published_splits/test.jsonl
```

---

## Reproducing the Dataset

The full construction pipeline lives under [pipeline/](pipeline/) and is organized into five stages mirroring Section 3 of the paper:

| Stage | Module | Paper § |
|---|---|---|
| 1. Ontology matching | `pipeline/1_ontology_matching/` | §3.1 |
| 2. Subgraph extraction | `pipeline/2_subgraph_extraction/` | §3.2 |
| 3. Answer coverage | `pipeline/3_answer_coverage/` | §3.3 |
| 4. Path annotation | `pipeline/4_path_annotation/` | §3.4 |
| 5. Splits | `pipeline/5_splits/` | §4 |

Each stage reads from the previous stage's output under `data/`. A full re-run from HotpotQA + Text2KGBench requires querying Wikidata extensively (~24 h) and a local LLM for Tier-3 fuzzy matching. For a fast end-to-end smoke test on a small slice, use the demo:

```bash
python demo/run_pipeline_demo.py
```

This downloads a small HotpotQA sample, runs the pipeline locally, and writes outputs under `demo/_workspace/`. It rebuilds its caches automatically — no Zenodo download needed.

---

## Repository Structure

```
.
├── README.md                  # this file
├── LICENSE                    # MIT (code)
├── LICENSE-DATASET            # CC BY-SA 4.0 (data)
├── CITATION.cff
├── requirements.txt
│
├── pipeline/                  # construction pipeline (5 stages)
├── baselines/                 # evaluation scripts + released predictions
│   └── results/
│       ├── oracle/            # oracle KG predictions for all 4 models
│       └── closed_book/       # closed-book predictions for all 4 models
├── demo/                      # end-to-end mini-pipeline demo for reviewers
├── ontologies/                # 10 Text2KGBench ontologies (post-extension)
├── validation/                # dataset validation scripts
├── utils/                     # shared helpers
└── docs/
    └── DATASET_CARD.md        # full schema specification
```

---

## Citation

If you use MetaHotpotQA, please cite the paper **and** the Zenodo deposit:

```bibtex
@inproceedings{metahotpotqa2026,
  title     = {MetaHotpotQA: Shifting Schemaless KGQA into Domain-Aware Knowledge Graphs},
  author    = {TODO},
  booktitle = {Proceedings of the 25th International Semantic Web Conference (ISWC), Resources Track},
  year      = {2026}
}

@dataset{metahotpotqa_zenodo_2026,
  title     = {MetaHotpotQA v1.0},
  author    = {TODO},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20071935}
}
```

---

## License

- **Dataset** (splits, subgraphs): [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) — inherits the share-alike terms of HotpotQA.
- **Underlying Wikidata content**: [CC0](https://creativecommons.org/publicdomain/zero/1.0/).
- **Ontologies**: as released by [Text2KGBench](https://arxiv.org/abs/2308.02357), CC BY 4.0.
- **Code**: MIT — see [LICENSE](LICENSE).

---

## Acknowledgements

MetaHotpotQA builds on [HotpotQA](https://hotpotqa.github.io/) (Yang et al., 2018), [Text2KGBench](https://arxiv.org/abs/2308.02357) (Mihindukulasooriya et al., 2023), and [Wikidata](https://www.wikidata.org/). We thank these communities for releasing high-quality open resources.
