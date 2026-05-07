# MetaHotpotQA — Dataset Card

## Overview

MetaHotpotQA is a schema-aware multi-hop Knowledge Graph Question Answering (KGQA) benchmark.
Each question is paired with a per-question Wikidata subgraph extracted under the constraints of
one or more domain ontologies from Text2KGBench, and annotated with a verified reasoning path
and a difficulty label.

- **Questions:** 12,056 (human-authored, sourced from HotpotQA)
- **Domains:** 10 (Movie, Music, Sport, Book, Military, Computer, Space, Politics, Nature, Culture)
- **KG:** Wikidata
- **Task type:** Multi-hop KGQA
- **Resource type:** Dataset
- **License:** CC BY-SA 4.0

---

## Splits

| Split | Questions |
|-------|----------:|
| train |     8,439 |
| dev   |     1,809 |
| test  |     1,808 |
| **Total** | **12,056** |

Splits are stratified over `difficulty × reasoning_type` with a fixed random seed (42).

---

## Field Descriptions

Each record in the JSONL files contains the following fields.

### Core fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | `string` | Original HotpotQA question ID. Use this to join with the upstream HotpotQA corpus to recover the Wikipedia supporting facts and the original `context` field. |
| `question` | `string` | Natural language question, sourced verbatim from HotpotQA. |
| `answer` | `string` | Gold answer string, sourced verbatim from HotpotQA. |
| `answer_node_qid` | `string` | Wikidata QID of the answer entity (e.g. `"Q231694"`). |

### KG grounding fields

| Field | Type | Description |
|-------|------|-------------|
| `entity_qids` | `list[string]` | Wikidata QIDs of the seed entities, derived from the HotpotQA supporting facts via Wikipedia–Wikidata alignment. |
| `matched_ontologies` | `list[object]` | Ontologies matched for this question. Each entry has: `ont_id` (ontology identifier, e.g. `"ont_2_music"`) and `matched_qids` (list of seed QIDs that triggered the match). |
| `supporting_paths` | `list[object]` | Reasoning paths recovered in the per-question Wikidata subgraph. Each entry has: `length` (number of hops) and `triples` (list of `[subject_qid, predicate_pid, object_qid]`). Always non-empty for `traversal` records (100%), recovered for 60.4% of `entity_selection` records, always empty (`[]`) for `property_comparison` records (no path traversal applies by design). This field is required to construct the oracle KG context used in the paper and to classify records into `oracle_full` vs `oracle_partial` regimes. |

### Difficulty and reasoning fields

| Field | Type | Values | Description |
|-------|------|--------|-------------|
| `reasoning_type` | `string` | `bridge`, `comparison` | HotpotQA question type. `bridge` requires chaining through an intermediate entity; `comparison` requires comparing a shared property of two seed entities. |
| `difficulty` | `string` | `entity_selection`, `traversal`, `property_comparison` | Difficulty category assigned by the pipeline. `entity_selection`: answer is one of the seed entities, path connects seeds. `traversal`: answer is disjoint from seeds, BFS path required. `property_comparison`: all comparison questions, no traversal chain exists. |
| `match_tier` | `int` | `1`, `2`, `3` | Confidence of the answer grounding during construction. `1` = exact string match; `2` = normalized match (lowercased, punctuation stripped); `3` = fuzzy match confirmed by LLM. Use this field to filter for higher-confidence subsets. |
| `strategy_used` | `int` | — | Internal BFS strategy index used by the path finder to recover the supporting path. Provided for reproducibility. |

---

## Difficulty Distribution (full dataset)

| Difficulty | Count | % |
|---|---:|---:|
| `entity_selection` | 6,970 | 57.8% |
| `traversal` | 3,407 | 28.3% |
| `property_comparison` | 1,679 | 13.9% |

---

## Domain Distribution

Questions can match multiple domains simultaneously (multi-label).
Counts are per question–domain occurrence.

| Domain | Occurrences |
|--------|------------:|
| Book | 7,647 |
| Movie | 7,454 |
| Music | 1,715 |
| Nature | 1,161 |
| Politics | 591 |
| Sport | 576 |
| Military | 557 |
| Culture | 146 |
| Space | 83 |
| Computer | 54 |

---

## Omitted Fields

The following fields present in the internal pipeline output are **not included** in this release:

| Field | Reason |
|-------|--------|
| `context` | Original HotpotQA Wikipedia supporting facts. Recoverable by joining on `id` with the [HotpotQA](https://hotpotqa.github.io/) training split. |
| `entities` | Per-entity detail (title, sentence index, QID, classes) for each seed entity. Recoverable by joining on `id` with HotpotQA and re-running Wikidata entity resolution. |
| `answer_aliases` | Was always identical to `[answer]` — the planned Wikidata alias extension was not applied. |
| `path_found` | Boolean flag, redundant with `supporting_paths` being non-empty. |
| `split` | Redundant with the file name. |

---

## Per-Question Subgraphs

Each question is accompanied by a per-question Wikidata subgraph (one JSON file per question,
named by `id`). The subgraph contains all nodes and edges downloaded during ontology-driven
extraction and is the source from which the oracle KG context is constructed for evaluation.
Subgraphs are distributed separately from the JSONL splits.

---

## Loading the Dataset

```python
import json

def load_split(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

train = load_split("train.jsonl")
dev   = load_split("dev.jsonl")
test  = load_split("test.jsonl")
```

To reconstruct the oracle KG context used in the paper, refer to the released pipeline code
and the oracle context construction procedure described in Section 4.1 of the paper.

---

## Upstream Datasets

| Resource | Role |
|----------|------|
| [HotpotQA](https://hotpotqa.github.io/) (Yang et al., 2018) | Source of questions and supporting facts |
| [Text2KGBench](https://arxiv.org/abs/2308.02357) (Mihindukulasooriya et al., 2023) | Source of the 10 domain ontologies |
| [Wikidata](https://www.wikidata.org/) | KG from which subgraphs are extracted |

---

## License

- **Dataset:** [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) — inherits the share-alike terms of HotpotQA.
- **Wikidata content:** [CC0](https://creativecommons.org/publicdomain/zero/1.0/)
- **Construction code:** MIT

---

## Citation

```bibtex
@inproceedings{metahotpotqa2025,
  title     = {MetaHotpotQA: Shifting Schemaless KGQA into Domain-Aware Knowledge Graphs},
  author    = {TODO},
  booktitle = {TODO},
  year      = {2025}
}
```
