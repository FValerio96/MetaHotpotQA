import os
import requests
import time
from typing import Dict, List, Iterable, Set

ENDPOINT = "https://www.wikidata.org/w/api.php"


def _user_agent() -> str:
    """Build the User-Agent string. Wikidata requires a contact email per its
    UA policy (https://meta.wikimedia.org/wiki/User-Agent_policy). Set the
    WIKIDATA_CONTACT environment variable to your email before running the
    pipeline.
    """
    contact = os.environ.get("WIKIDATA_CONTACT", "").strip()
    if not contact:
        raise RuntimeError(
            "WIKIDATA_CONTACT env var is required (set it to a contact email "
            "or URL identifying you, per Wikidata's User-Agent policy).")
    return f"MetaHotpotQA/1.0 (https://github.com/FValerio96/MetaHotpot; mailto:{contact})"


HEADERS = {"User-Agent": _user_agent()}

# Retry configuration
MAX_RETRIES = 10
BASE_WAIT = 5.0   # exponential, capped at MAX_WAIT
MAX_WAIT = 60.0   # cap per-attempt wait (server-side maxlag is overload, not throttling — keep retrying patiently)


def chunked(items: Iterable[str], n: int) -> List[List[str]]:
    items = list(items)
    return [items[i:i + n] for i in range(0, len(items), n)]


def _request_with_retry(params: dict, max_retries: int = MAX_RETRIES) -> dict:
    """
    Run a GET request with retry and exponential backoff.
    Handles 429 errors, timeouts, and connection issues.
    """
    def _wait(reason: str, attempt: int, multiplier: float = 1.0) -> None:
        wait_time = min((2 ** attempt) * BASE_WAIT * multiplier, MAX_WAIT) + (time.time() % 1)
        print(f"    [{reason}] Waiting {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})")
        time.sleep(wait_time)

    for attempt in range(max_retries):
        try:
            resp = requests.get(ENDPOINT, params=params, headers=HEADERS, timeout=30)

            if resp.status_code == 429:
                _wait("429", attempt)
                continue

            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict) and data.get("error", {}).get("code") == "maxlag":
                _wait("maxlag", attempt)
                continue
            return data

        except requests.exceptions.Timeout:
            _wait("Timeout", attempt)
        except requests.exceptions.ConnectionError:
            _wait("ConnectionError", attempt, multiplier=2.0)
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                _wait("429", attempt)
            else:
                raise

    raise Exception(f"Max retries ({max_retries}) exceeded")


def wikidata_get_entities(
    qids: Iterable[str],
    props: str = "labels|descriptions|claims",
    batch_size: int = 50,
) -> Dict[str, dict]:
    """
    Wrapper around wbgetentities with batching and automatic retry.
    Returns a dict mapping QID -> entity JSON with all its claims.
    """
    out: Dict[str, dict] = {}
    for batch in chunked(set(qids), batch_size):
        params = {
            "action": "wbgetentities",
            "format": "json",
            "ids": "|".join(batch),
            "props": props,
            "maxlag": "5",  # respect Wikidata's maxlag
        }
        data = _request_with_retry(params)
        out.update(data.get("entities", {}))
    return out

# NOTE: "targets" is the technical name for the values of a Wikidata triple
# SUBJECT --PID--> TARGET
def extract_targets(claims: dict, pid: str) -> List[str]:
    """
    Extract the QIDs that act as values of a triple via the property pid.
    """
    vals: List[str] = []
    for c in claims.get(pid, []):
        dv = c.get("mainsnak", {}).get("datavalue", {})
        if dv.get("type") == "wikibase-entityid":
            vals.append("Q" + str(dv["value"]["numeric-id"]))
    return vals


def fetch_types(qids: Iterable[str]) -> Dict[str, Dict[str, List[str]]]:
    """
    For a list of QIDs, download only the P31/P279 claims.
    Returns: { qid: {"instance_of": [...], "subclass_of": [...]} }.
    """
    entities = wikidata_get_entities(qids, props="claims")
    typed: Dict[str, Dict[str, List[str]]] = {}
    for qid, ent in entities.items():
        if ent.get("missing") is not None:
            continue
        claims = ent.get("claims", {})
        typed[qid] = {
            "instance_of": extract_targets(claims, "P31"),
            "subclass_of": extract_targets(claims, "P279"),
        }
    return typed


def download_node_with_ontology_props(
    qid: str,
    property_list: List[Dict],  # objects with keys: pid, label, domain, range
) -> dict:
    """
    Given a QID that is in the domain of at least one property in the ontology,
    its qid_classes (instance_of or subclass_of of the node), and a property
    list from the ontology to filter against qid_classes, download the node
    from Wikidata and return only the specified properties, filtering the
    targets by 'range'.

    - If range is [] -> keep all targets.
    - If range is non-empty -> keep targets that:
        * have type (P31/P279) in range, OR
        * are themselves in range.
    """


    entities = wikidata_get_entities([qid])  # wbgetentities batch wrapper, here used for a single entity
    entity = entities.get(qid)  # extract the entity from the returned dict
    if not entity or entity.get("missing") is not None:
        raise ValueError(f"Entity {qid} not found on Wikidata.")

    labels = entity.get("labels", {})
    descriptions = entity.get("descriptions", {})
    claims = entity.get("claims", {})  # claims are the triples the entity participates in

    # 0-hop node construction
    node = {
        "qid": qid,
        "label": labels.get("en", {}).get("value"),
        "description": descriptions.get("en", {}).get("value"),
        "instance_of": extract_targets(claims, "P31"),
        "subclass_of": extract_targets(claims, "P279"),
        "props": {},  # pid -> list of filtered targets
    }

    # 1) First extract all candidate targets for properties with a non-empty
    #    range, so we can fetch their types in a SINGLE call.
    qids_to_typecheck: Set[str] = set()  # QIDs for which we need types
    pid_to_targets_raw: Dict[str, List[str]] = {}  # pid -> raw targets

    qid_classes = set(node["instance_of"]) | set(node["subclass_of"])
    # keep properties for which the QID's type is in their domain
    # (type = instance_of + subclass_of)
    filtered_props = [
        p for p in property_list
        if p["domain"] in qid_classes
    ]
    for prop in filtered_props:
        pid = prop["pid"]
        targets = extract_targets(claims, pid)
        pid_to_targets_raw[pid] = targets  # for each pid, store raw targets

        if prop.get("range"):  # non-empty range -> we need types
            qids_to_typecheck.update(targets)  # collect all targets that need range filtering

    # 2) Fetch types of the targets that must be filtered by range
    types_map: Dict[str, Dict[str, List[str]]] = {}
    if qids_to_typecheck:
        types_map = fetch_types(qids_to_typecheck)  # look up instance_of of target nodes for range checks

    # 3) Apply the range filter and fill node["props"]
    for prop in filtered_props:
        pid = prop["pid"]
        allowed_range: List[str] = prop.get("range", [])
        raw_targets = pid_to_targets_raw.get(pid, [])

        if not raw_targets:
            continue

        if not allowed_range:
            # empty range -> keep all targets
            filtered = raw_targets
        else:
            allowed_set = set(allowed_range)
            filtered: List[str] = []
            for tgt in raw_targets:
                # if the target itself equals a range root, keep it
                if tgt in allowed_set:
                    filtered.append(tgt)
                    continue

                tinfo = types_map.get(tgt, {})
                inst = set(tinfo.get("instance_of", []))
                subc = set(tinfo.get("subclass_of", []))

                # keep the target if one of its types matches the range
                if inst & allowed_set or subc & allowed_set:
                    filtered.append(tgt)

        if filtered:
            node["props"][pid] = filtered

    return node

def fetch_node_text_info(qids: Iterable[str]) -> Dict[str, dict]:
    """
    For each QID, return only textual info (0-hop):
    label, description, instance_of, subclass_of.
    """
    entities = wikidata_get_entities(qids, props="labels|descriptions|claims")
    out = {}
    for qid, ent in entities.items():
        if ent.get("missing") is not None:
            continue
        claims = ent.get("claims", {})
        out[qid] = {
            "qid": qid,
            "label": ent.get("labels", {}).get("en", {}).get("value"),
            "description": ent.get("descriptions", {}).get("en", {}).get("value"),
            "instance_of": extract_targets(claims, "P31"),
            "subclass_of": extract_targets(claims, "P279"),
        }
    return out



def extend_neighbors_with_concepts(
    node: dict,
    concepts_cfg: Dict[str, List[str]],
) -> Dict[str, dict]:
    """
    node: output of download_node_with_ontology_props(...)
    concepts_cfg: e.g. {"Q5": ["P166", "P1411"], "Q618779": ["P585", "P1686"], ...}

    Returns:
        { neighbor_qid -> {
            "qid": ...,
            "label": ...,
            "description": ...,
            "instance_of": [...],
            "subclass_of": [...],
            "extra_props": { pid -> [targets] }
        }}
    """
    # 1) Collect all neighbors (QIDs) of the node
    neighbor_qids: Set[str] = set()
    for targets in node.get("props", {}).values():
        neighbor_qids.update(targets)

    if not neighbor_qids:
        return {}

    # 2) Download neighbor info (labels, descriptions, claims) in a SINGLE
    #    call. Types are derived from this same response.
    entities = wikidata_get_entities(neighbor_qids, props="labels|descriptions|claims")
    # 3) Extract neighbor types (P31/P279) from 'entities' without
    #    additional Wikidata calls.
    types_map = {
        qid: {
            "instance_of": extract_targets(ent.get("claims", {}), "P31"),
            "subclass_of": extract_targets(ent.get("claims", {}), "P279")
        }
        for qid, ent in entities.items()
        if ent.get("missing") is None
    }

    extended: Dict[str, dict] = {}
    concepts_qids: Set[str] = set(concepts_cfg.keys())

    for qid, ent in entities.items():
        if ent.get("missing") is not None:
            continue

        tinfo = types_map.get(qid, {})
        inst = set(tinfo.get("instance_of", []))
        subc = set(tinfo.get("subclass_of", []))
        classes = inst | subc

        # 4) Check whether the neighbor belongs to one of the extendable concepts
        matched_concepts = classes & concepts_qids
        if not matched_concepts:
            continue

        labels = ent.get("labels", {})
        descriptions = ent.get("descriptions", {})
        claims = ent.get("claims", {})

        # 5) Merge follow_pids of all matching concepts
        follow_pids: Set[str] = set()
        for cq in matched_concepts:
            follow_pids.update(concepts_cfg[cq])

        # 6) Extract only the targets for those pids
        extra_props: Dict[str, List[str]] = {}
        for pid in follow_pids:
            targets = extract_targets(claims, pid)
            if targets:
                extra_props[pid] = targets

        if extra_props:
            extended[qid] = {
                "qid": qid,
                "label": labels.get("en", {}).get("value"),
                "description": descriptions.get("en", {}).get("value"),
                "instance_of": list(inst),
                "subclass_of": list(subc),
                "extra_props": extra_props,
            }

    return extended


def preload_ontology_concepts(
    ontology_concepts: Set[str],
    node_cache: Dict[str, dict],
) -> None:
    """
    Ensure all ontology concepts are present in node_cache as 0-hop nodes
    (label, description, P31, P279).
    """
    missing = ontology_concepts - set(node_cache.keys())
    if not missing:
        return
    concept_nodes = fetch_node_text_info(missing)
    merge_into_cache(concept_nodes, node_cache)


def download_node_general(
    qid: str,
    classes: List[str],
    property_list: List[Dict],
    concepts_cfg: Dict[str, List[str]],
) -> Dict[str, dict]:
    qid_classes = set(classes)
    domains = {p["domain"] for p in property_list}
    if qid_classes & domains:
        node = download_node_with_ontology_props(qid, property_list)
        neighbor_qids = {t for targets in node["props"].values() for t in targets}
        neighbor_nodes = fetch_node_text_info(neighbor_qids)
        extended_neighbors = extend_neighbors_with_concepts(node, concepts_cfg)
        extra_qids = {t for ext in extended_neighbors.values()
                        for targets in ext["extra_props"].values()
                        for t in targets}
        extra_qids -= ({qid} | neighbor_qids)
        extra_nodes = fetch_node_text_info(extra_qids)
        nodes = {qid: node, **neighbor_nodes, **extra_nodes}
        for q, ext in extended_neighbors.items():
            nodes[q] = merge_node_dicts(nodes.get(q, {}), ext)
        return normalize_nodes_qid_field(nodes)
    else:
        nodes = download_node_1hop(qid)
        return normalize_nodes_qid_field(nodes)
    


def download_node_1hop(qid: str) -> Dict[str, dict]:
    entities = wikidata_get_entities([qid], props="labels|descriptions|claims")
    ent = entities.get(qid)
    if not ent or ent.get("missing") is not None:
        raise ValueError(f"Entity {qid} not found.")

    labels = ent.get("labels", {})
    descriptions = ent.get("descriptions", {})
    claims = ent.get("claims", {})

    # central node
    main_node = {
        "qid": qid,
        "label": labels.get("en", {}).get("value"),
        "description": descriptions.get("en", {}).get("value"),
        "instance_of": extract_targets(claims, "P31"),
        "subclass_of": extract_targets(claims, "P279"),
        "props": {},
    }

    neighbor_qids: Set[str] = set()
    props: Dict[str, List[str]] = {}

    for pid in claims.keys():
        targets = extract_targets(claims, pid)
        if targets:
            props[pid] = targets
            neighbor_qids.update(targets)

    main_node["props"] = props

    # 0-hop for all neighbors
    neighbor_nodes = fetch_node_text_info(neighbor_qids)

    # return a dict like in the ontology branch
    nodes: Dict[str, dict] = {qid: main_node, **neighbor_nodes}
    return normalize_nodes_qid_field(nodes)




def ensure_connectivity(
    ex_ids: List[str],
    edges: List[tuple],
) -> bool:
    """
    Return True if all qids in ex_ids belong to the same connected component
    of the graph defined by edges, otherwise False.
    """
    adj = build_local_adj(edges)
    return is_connected(ex_ids, adj)


SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
SPARQL_HEADERS = {"User-Agent": _user_agent()}

def get_min_path_from_wikidata(
    source: str,
    target: str,
    max_len: int = 4,
) -> tuple[List[str], List[tuple]]:
    """
    TODO: implement via SPARQL property paths or BFS over the API.
    Should return:
      - path_qids: [q_source, ..., q_target]
      - path_edges: [(q_i, pid, q_{i+1}), ...]
    Currently a stub.
    """
    raise NotImplementedError("get_min_path_from_wikidata must be implemented via SPARQL/BFS.")



def build_kg_for_example(
    example: dict,
    property_list: List[Dict],
    ontology_concepts: Set[str],
    concepts_cfg: Dict[str, List[str]],
    node_cache: Dict[str, dict],
) -> tuple[Dict[str, dict], List[tuple], bool]:
    local_nodes: Dict[str, dict] = {}
    local_edges: List[tuple] = []

    for ent in example["ex_qids"]:
        qid = ent["qid"]
        classes = ent.get("classes", [])
        if qid in node_cache:
            local_nodes[qid] = node_cache[qid]
        else:
            nodes_q = download_node_general(qid, classes, property_list, concepts_cfg)
            merge_into_cache(nodes_q, node_cache)
            local_nodes.update(nodes_q)

        # TYPE edges to the ontology's concepts
        local_edges.extend(
            add_type_edges_for_entity(qid, classes, ontology_concepts)
        )

    ex_ids = [e["qid"] for e in example["ex_qids"]]
    connected = ensure_connectivity(ex_ids, local_edges)
    return local_nodes, local_edges, connected


def merge_node_dicts(a: dict, b: dict) -> dict:
    out = dict(a)
    # merge base lists
    for key in ["instance_of", "subclass_of"]:
        la = set(out.get(key, []))
        lb = set(b.get(key, []))
        out[key] = list(la | lb)
    # merge props / extra_props
    for field in ["props", "extra_props"]:
        pa = out.get(field, {})
        pb = b.get(field, {})
        merged = {k: list(set(pa.get(k, []) + pb.get(k, []))) for k in set(pa) | set(pb)}
        if merged:
            out[field] = merged
    # label/description: prefer the existing one, otherwise take from b
    for key in ["label", "description"]:
        if not out.get(key) and b.get(key):
            out[key] = b[key]
    return out


def add_type_edges_for_entity(qid, classes, ontology_concepts):
    return list({
        (qid, "TYPE", c)
        for c in classes
        if c in ontology_concepts
    })



def merge_into_cache(local_nodes: Dict[str, dict], node_cache: Dict[str, dict]) -> None:
    for q, n in local_nodes.items():
        if q in node_cache:
            node_cache[q] = merge_node_dicts(node_cache[q], n)
        else:
            node_cache[q] = n


def build_local_adj(edges: List[tuple]) -> Dict[str, Set[str]]:
    adj: Dict[str, Set[str]] = {}
    for s, _, o in edges:
        adj.setdefault(s, set()).add(o)
        adj.setdefault(o, set()).add(s)
    return adj


def _bfs(start: str, adj: Dict[str, Set[str]]) -> Set[str]:
    seen = {start}
    queue = [start]
    while queue:
        u = queue.pop(0)
        for v in adj.get(u, []):
            if v not in seen:
                seen.add(v)
                queue.append(v)
    return seen


def is_connected(ex_ids: List[str], adj: Dict[str, Set[str]]) -> bool:
    if not ex_ids:
        return True
    seen = _bfs(ex_ids[0], adj)
    return all(q in seen for q in ex_ids)


def path_exists(u: str, v: str, adj: Dict[str, Set[str]]) -> bool:
    return v in _bfs(u, adj) if u in adj and v in adj else False

def normalize_nodes_qid_field(nodes: Dict[str, dict]) -> Dict[str, dict]:
    for qid, n in nodes.items():
        if isinstance(n, dict) and "qid" not in n:
            n["qid"] = qid
    return nodes

def extract_prop_edges(nodes: Dict[str, dict]) -> List[dict]:
    """
    Convert props / extra_props into edges:
    each edge = {source, target, type} where type is the PID (e.g. "P57").
    Keeps only edges whose target is present in nodes.
    """
    edges: List[dict] = []
    qid_set = set(nodes.keys())

    for node in nodes.values():
        if not isinstance(node, dict) or "qid" not in node:
            continue
        src = node["qid"]
        for field in ("props", "extra_props"):
            for pid, targets in node.get(field, {}).items():
                for tgt in targets:
                    if tgt in qid_set:
                        edges.append({
                            "source": src,
                            "target": tgt,
                            "type": pid,  # in Neo4j this becomes the relationship type
                        })
    return edges


def edges_to_dict_list(type_edges: List[tuple]) -> List[dict]:
    """
    Convert TYPE edges (qid, "TYPE", concept_qid) into dicts for JSON.
    """
    return [
        {"source": s, "target": t, "type": rel}
        for (s, rel, t) in type_edges
    ]




