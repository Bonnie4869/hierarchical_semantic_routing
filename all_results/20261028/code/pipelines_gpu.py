# Cached CUDA Sinkhorn variant; bounded reference and GPU caches, optional profiling.
# CUDA Sinkhorn variant; centroid and file/feature processing remain on CPU.
"""Three centroid-based greedy DDC pipelines using existing reference artifacts.

Commands: prepare (disk store), queries (spaCy features with global TF-IDF), compare.
Only load pickle files you trust. No feature training or reference regeneration.

Methods:
  centroid: all current-level candidates -> centroid ranking -> Top-K2.
  centroid_wmd: centroid Top-K1 -> centroid_similarity * WMD_similarity -> Top-K2.
  centroid_sinkhorn: centroid Top-K1 -> centroid_similarity * Sinkhorn_similarity
  -> Top-K2.

Every item in Top-K2 is expanded at the next DDC level. Thus DL2 searches the
union of all direct children of DL1 Top-K2, and DL3 searches the union of all
direct children of DL2 Top-K2. The first Top-K2 item is still reported as the
level's Top-1 prediction. WMD and Sinkhorn keep token embeddings at their
original positions. Feature weights are normalized and used only as OT marginals.
"""

import argparse
import csv
import gc
import heapq
import importlib.util
import json
import math
import pickle
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
from collections import OrderedDict, defaultdict
from functools import lru_cache
import weakref

_PROFILE = defaultdict(float)
_PROFILE_PHASES = False
_GPU_CACHE = OrderedDict()
_GPU_CACHE_BYTES = 0
_GPU_CACHE_LIMIT = 256 * 1024 * 1024


def _gpu_distribution(a, context):
    global _GPU_CACHE_BYTES
    torch, device, dtype, _ = context
    key = (id(a[0]), id(a[1]))
    cached = _GPU_CACHE.pop(key, None)
    if cached is not None:
        if cached[0]() is a[0] and cached[1]() is a[1]:
            _GPU_CACHE[key] = cached
            _PROFILE["gpu_cache_hits"] += 1
            return cached[2]
        _GPU_CACHE_BYTES -= cached[3]
    _PROFILE["gpu_cache_misses"] += 1
    weights = torch.as_tensor(a[0], device=device, dtype=dtype)
    vectors = torch.as_tensor(a[1], device=device, dtype=dtype)
    weights = weights / weights.sum()
    size = weights.numel() * weights.element_size() + vectors.numel() * vectors.element_size()
    if size <= _GPU_CACHE_LIMIT:
        while _GPU_CACHE and _GPU_CACHE_BYTES + size > _GPU_CACHE_LIMIT:
            _, entry = _GPU_CACHE.popitem(last=False)
            _GPU_CACHE_BYTES -= entry[3]
        _GPU_CACHE[key] = (weakref.ref(a[0]), weakref.ref(a[1]), (weights, vectors), size)
        _GPU_CACHE_BYTES += size
    return weights, vectors


LAYERS = (1, 2, 3)
WEIGHTS = (0.12, 0.21, 0.67)


def label(value):
    text = str(value).strip()
    try:
        number = float(text)
        if math.isfinite(number) and number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    if not text or text.lower() in ("nan", "none"):
        raise ValueError("Missing class label")
    return text


def module_at(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_frame(path):
    import pandas as pd

    return (
        pd.read_csv(path)
        if Path(path).suffix.lower() == ".csv"
        else pd.read_excel(path)
    )


def read_hierarchy(path):
    """Training sampling writes separate DL1/DL2/DL3 sheets; read them all."""
    import pandas as pd

    required = ["DL1", "DL2", "DL3"]
    if Path(path).suffix.lower() == ".csv":
        frames = [pd.read_csv(path)]
    else:
        frames = list(pd.read_excel(path, sheet_name=None).values())
    frames = [
        frame[required] for frame in frames if set(required).issubset(frame.columns)
    ]
    if not frames:
        raise ValueError("Training hierarchy has no sheet containing DL1/DL2/DL3")
    return pd.concat(frames, ignore_index=True).drop_duplicates()


def reference_parents(keys, hierarchy):
    """Metadata defines existing nodes. Never create theoretical DDC nodes."""
    parents = {}
    key_set = set(keys)
    for level in (2, 3):
        for child, parent in (
            hierarchy[[f"DL{level}", f"DL{level-1}"]]
            .dropna()
            .itertuples(index=False, name=None)
        ):
            key, value = (level, label(child)), label(parent)
            if key not in key_set:
                continue
            if key in parents and parents[key] != value:
                raise ValueError(f"Multiple parents for {key}")
            parents[key] = value
    for level, name in keys:
        if level == 1:
            continue
        parent = parents.get((level, name))
        if parent is None:
            raise ValueError(
                f"Reference node {(level, name)} has no parent mapping across training sheets; verify matching training/reference versions"
            )
        if (level - 1, parent) not in key_set:
            raise ValueError(
                f"Reference node {(level, name)} has parent {(level-1, parent)} absent from metadata; verify matching reference files"
            )
    return parents


def prepare(args):
    """One-off serialization only. Category mode holds one node at a time."""
    target = Path(args.store)
    if target.exists():
        raise FileExistsError(f"{target}: choose a new store filename")
    refs = Path(args.refs)
    with (refs / "reference_metadata.pkl").open("rb") as f:
        metadata = pickle.load(f)
    matrices = [
        np.load(refs / f"L{s}_reference.npy", mmap_mode="r", allow_pickle=False)
        for s in LAYERS
    ]
    if any(
        m.ndim != 2 or len(m) != len(metadata) or m.shape != matrices[0].shape
        for m in matrices
    ):
        raise ValueError("Reference matrices and metadata shapes disagree")
    keys = [
        (int(m.get("ddc_level", m.get("level"))), label(m["name"])) for m in metadata
    ]
    if len(set(keys)) != len(keys) or any(k[0] not in LAYERS for k in keys):
        raise ValueError("Duplicate/invalid reference keys")
    hierarchy = read_hierarchy(args.hierarchy)
    parents = reference_parents(keys, hierarchy)
    bows = None
    if args.bows:
        # A legacy list pickle must be deserialized once; never used in inference.
        with open(args.bows, "rb") as f:
            items = pickle.load(f)
        bows = {(int(b["ddc_level"]), label(b["name"])): b["semantic"] for b in items}
        if len(bows) != len(items) or set(bows) != set(keys):
            raise ValueError("Bow identities do not match metadata")
        del items
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".building")
    if temporary.exists():
        raise FileExistsError(temporary)
    db = sqlite3.connect(temporary)
    try:
        db.execute(
            "CREATE TABLE nodes (id INTEGER PRIMARY KEY, level INTEGER, name TEXT, parent TEXT, centroid BLOB, bow BLOB, UNIQUE(level,name))"
        )
        db.execute("CREATE INDEX child_lookup ON nodes(level,parent)")
        for i, (level, name) in enumerate(keys):
            if bows is not None:
                semantic = bows.pop((level, name))
            else:
                source = (
                    Path(args.categories)
                    / f"level{level}"
                    / f"category_{metadata[i]['name']}.pkl"
                )
                with source.open("rb") as f:
                    category = pickle.load(f)
                semantic = category["category_features"]
                del category
            centroid = np.asarray([m[i] for m in matrices], dtype=np.float64)
            if not np.isfinite(centroid).all():
                raise ValueError(f"Nonfinite centroid: {name}")
            db.execute(
                "INSERT INTO nodes VALUES (?,?,?,?,?,?)",
                (
                    i,
                    level,
                    name,
                    parents.get((level, name)),
                    pickle.dumps(centroid),
                    pickle.dumps(semantic),
                ),
            )
            del semantic, centroid
        db.commit()
    finally:
        db.close()
    temporary.rename(target)
    print(f"Stored {len(keys)} nodes: {target}")


def load_global_document_frequencies(path):
    """Merge DF statistics from all disjoint DL1 category files.

    The old query extractor builds one global Query IDF by summing the DF and
    document counts stored in every DL1 category_*.pkl. DL2/DL3 files must not
    be included because they contain the same training documents again.
    """
    from collections import Counter

    directory = Path(path)
    if (directory / "level1").is_dir():
        directory /= "level1"
    files = sorted(directory.glob("category_*.pkl"))
    if not files:
        raise FileNotFoundError(f"No category_*.pkl files found in {directory}")

    global_df = {level: Counter() for level in LAYERS}
    total_docs = 0
    for source in files:
        with source.open("rb") as f:
            category = pickle.load(f)
        if "df_counts" not in category or "total_docs" not in category:
            raise ValueError(f"Missing df_counts/total_docs in {source}")
        total_docs += int(category["total_docs"])
        for level in LAYERS:
            counts = category["df_counts"].get(
                level, category["df_counts"].get(str(level), {})
            )
            global_df[level].update(counts)

    if total_docs <= 0:
        raise ValueError("Global IDF corpus contains no documents")
    for level in LAYERS:
        for term, frequency in global_df[level].items():
            if frequency < 0 or frequency > total_docs:
                raise ValueError(
                    f"Invalid global DF at semantic L{level}: "
                    f"{term!r}={frequency}, total_docs={total_docs}"
                )
    return global_df, total_docs, directory


def query_tfidf(seq, document_frequencies, total_docs):
    """Return a sum-normalized global TF-IDF distribution for one query layer."""
    from collections import Counter

    if not seq:
        return {}
    counts = Counter(seq)
    length = len(seq)
    raw = {
        term: (count / length)
        * math.log((total_docs + 1) / (document_frequencies.get(term, 0) + 1))
        for term, count in counts.items()
    }
    normalizer = sum(raw.values())
    if normalizer <= 0:
        return {}
    return {
        "||".join(term) if isinstance(term, tuple) else term: weight / normalizer
        for term, weight in raw.items()
        if weight > 0
    }


def queries(args):
    """Reuse feature extraction and apply the old code's global Query TF-IDF."""

    builder = module_at(
        Path(args.training) / "feature_builder_node.py", "query_features"
    )
    global_df, total_docs, idf_directory = load_global_document_frequencies(
        args.idf_categories
    )
    frame = read_frame(args.input)
    if args.limit:
        frame = frame.head(args.limit)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as f:
        texts = frame[args.text_column].fillna("").astype(str)
        for i, doc in enumerate(builder.nlp.pipe(texts, batch_size=16)):
            _, terms = builder._extract_doc_features(doc)
            semantic = {}
            for level, seq in zip(LAYERS, terms):
                semantic[level] = query_tfidf(
                    seq, global_df[level], total_docs
                )
            row = frame.iloc[i]
            gold = {
                str(s): label(row[f"DL{s}"])
                for s in LAYERS
                if f"DL{s}" in frame and not __import__("pandas").isna(row[f"DL{s}"])
            }
            f.write(
                json.dumps(
                    {
                        "id": i,
                        "semantic": semantic,
                        "gold": gold,
                        "weighting": "global_tfidf",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(
        f"Prepared {len(frame)} global-TF-IDF queries using {total_docs} "
        f"training documents from {idf_directory}: {output}"
    )


class Store:
    def __init__(self, path):
        self.db = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
        self.db.execute("PRAGMA cache_size=-64")  # bounded 64 KiB SQLite page cache
        self.db.execute("PRAGMA mmap_size=0")
        self.db.execute("PRAGMA cache_size=-16384")
        self.distribution_cache = OrderedDict()
        self.distribution_cache_bytes = 0
        self.distribution_cache_limit = 512 * 1024 * 1024

    def candidates(self, level, parent, *, include_centroid=True):
        columns = "id,name,centroid" if include_centroid else "id,name"
        sql = f"SELECT {columns} FROM nodes WHERE level=? AND parent IS ? ORDER BY name"
        cursor = self.db.execute(sql, (level, parent))
        try:
            for row in cursor:
                if include_centroid:
                    node_id, name, blob = row
                    yield node_id, name, pickle.loads(blob)
                else:
                    node_id, name = row
                    yield node_id, name, None
        finally:
            cursor.close()

    def bow(self, node_id):
        return pickle.loads(
            self.db.execute("SELECT bow FROM nodes WHERE id=?", (node_id,)).fetchone()[
                0
            ]
        )

    def reference_sides(self, node_id, max_terms):
        key = (node_id, max_terms)
        cached = self.distribution_cache.pop(key, None)
        if cached is not None:
            self.distribution_cache[key] = cached
            _PROFILE["reference_cache_hits"] += 1
            return cached[0]
        _PROFILE["reference_cache_misses"] += 1
        start = time.perf_counter()
        raw = self.bow(node_id)
        result = sides(raw, self.embeddings, max_terms)
        del raw
        _PROFILE["reference_build_wall_seconds"] += time.perf_counter() - start
        size = sum(w.nbytes + v.nbytes for item in result if item is not None for w, v in [item])
        if size <= self.distribution_cache_limit:
            while self.distribution_cache and self.distribution_cache_bytes + size > self.distribution_cache_limit:
                _, (_, removed_size) = self.distribution_cache.popitem(last=False)
                self.distribution_cache_bytes -= removed_size
            self.distribution_cache[key] = (result, size)
            self.distribution_cache_bytes += size
        return result

    def release(self):
        # SQLite cache is bounded; releasing its pages on every node is unnecessary.
        pass

    def close(self):
        self.distribution_cache.clear()
        self.distribution_cache_bytes = 0
        self.db.close()


class Embeddings:
    def __init__(self, training, vectors, format_name):
        mod = module_at(
            Path(training) / "embed_reference_points_v2.py", "existing_embeddings"
        )
        data, kind, dim = mod.load_embeddings(vectors, format_name)
        self.wrapper = mod.EmbeddingWrapper(data, kind, dim)
        self.dim = dim
        self.term = lru_cache(maxsize=50000)(self.term)

    def term(self, term):
        parts = term if isinstance(term, tuple) else str(term).split("||")
        # Bypass unbounded _cache; the shared embedding model is not node data.
        vectors = [self.wrapper._get_vector_impl(word) for word in parts]
        valid = [v for v in vectors if np.isfinite(v).all() and np.linalg.norm(v) > 0]
        return np.mean(valid, axis=0) if valid else np.zeros(self.dim)


def side(bag, embeddings, max_terms=None):
    """Build one weighted semantic distribution for WMD/Sinkhorn.

    Token embeddings remain at their original positions. Feature weights are
    normalized to sum to one and used only as the OT marginal distribution.
    Reference weights are node-local TF-IDF; query weights are global TF-IDF
    computed by merging DF statistics from every DL1 training category.
    """
    weights, vectors = [], []
    items = list(bag.items())
    if max_terms is not None:
        # Deterministic top-weight pruning before vector lookup; ties use term text.
        items = heapq.nsmallest(
            max_terms, items, key=lambda item: (-float(item[1]), str(item[0]))
        )
    for term, weight in items:
        weight = float(weight)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("Feature weights must be finite and nonnegative")
        if weight == 0:
            continue
        vector = embeddings.term(term)
        if np.linalg.norm(vector) > 0:
            weights.append(weight)
            vectors.append(vector)
    if not weights:
        return None
    weights = np.asarray(weights, dtype=np.float64)
    weights /= weights.sum()
    return weights, np.asarray(vectors, dtype=np.float64)


def sides(semantic, embeddings, max_terms=None):
    return [
        side(
            semantic.get(s, semantic.get(str(s), {})),
            embeddings,
            max_terms,
        )
        for s in LAYERS
    ]


def centroids(distributions, dim):
    return np.asarray(
        [
            item[0] @ item[1] if item is not None else np.zeros(dim)
            for item in distributions
        ]
    )


def cosine_distance(query, candidate, weights):
    norms = np.linalg.norm(query, axis=1) * np.linalg.norm(candidate, axis=1)
    similarity = np.divide(
        np.sum(query * candidate, axis=1), norms, out=np.zeros(3), where=norms > 0
    )
    return float(np.dot(weights, 1 - np.clip(similarity, -1, 1)))


def sparse_tfidf_distance(query_semantic, reference_semantic, weights):
    """Old AG sparse-feature cosine, returned as a distance (smaller is better)."""
    layer_distances = []
    for s in LAYERS:
        query = query_semantic.get(s, query_semantic.get(str(s), {}))
        reference = reference_semantic.get(s, reference_semantic.get(str(s), {}))
        # JSON query tuples use ||; canonicalize legacy tuple keys from PKL.
        q = {
            ("||".join(k) if isinstance(k, tuple) else str(k)): float(v)
            for k, v in query.items()
        }
        r = {
            ("||".join(k) if isinstance(k, tuple) else str(k)): float(v)
            for k, v in reference.items()
        }
        q_norm = math.sqrt(sum(v * v for v in q.values()))
        r_norm = math.sqrt(sum(v * v for v in r.values()))
        similarity = (
            (sum(v * r.get(term, 0.0) for term, v in q.items()) / (q_norm * r_norm))
            if q_norm > 0 and r_norm > 0
            else 0.0
        )
        layer_distances.append(1.0 - max(-1.0, min(1.0, similarity)))
    return float(np.dot(weights, layer_distances))


GROUND_COSTS = (
    "euclidean",
    "normalized_euclidean",
    "cosine",
    "normalized_squared_euclidean",
)


def ground_cost_matrix(av, bv, ground_cost="euclidean"):
    """Build the term-to-term OT ground-cost matrix.

    ``normalized_squared_euclidean`` follows the common POT tutorial convention:
    compute squared Euclidean costs from the original embeddings, then divide
    each query/reference cost matrix by its own maximum value.
    """
    av = np.asarray(av, dtype=np.float64)
    bv = np.asarray(bv, dtype=np.float64)
    if av.ndim != 2 or bv.ndim != 2 or av.shape[1] != bv.shape[1]:
        raise ValueError("Embedding matrices must be 2-D with matching dimensions")
    if ground_cost not in GROUND_COSTS:
        raise ValueError(f"Invalid ground cost: {ground_cost}")

    if ground_cost in ("normalized_euclidean", "cosine"):
        av_norms = np.linalg.norm(av, axis=1, keepdims=True)
        bv_norms = np.linalg.norm(bv, axis=1, keepdims=True)
        if np.any(av_norms <= 0) or np.any(bv_norms <= 0):
            raise ValueError("Normalized ground costs require nonzero embeddings")
        left = av / av_norms
        right = bv / bv_norms
    else:
        left, right = av, bv

    if ground_cost == "cosine":
        cost = 1.0 - left @ right.T
        # Roundoff may produce values just outside the theoretical [0, 2] range.
        np.clip(cost, 0.0, 2.0, out=cost)
        return cost

    # Squared Euclidean distance without n*m*embedding_dimension broadcasting.
    cost = left @ right.T
    cost *= -2
    cost += np.sum(left * left, axis=1)[:, None]
    cost += np.sum(right * right, axis=1)[None, :]
    np.maximum(cost, 0, out=cost)

    if ground_cost == "normalized_squared_euclidean":
        maximum = float(np.max(cost))
        if maximum > 0:
            cost /= maximum
        return cost

    # euclidean and normalized_euclidean both use the ordinary L2 norm.
    np.sqrt(cost, out=cost)
    return cost


# GPU settings are initialized by worker; CPU retains the original NumPy solver.
_OT_CONTEXT = None


def configure_ot_device(args):
    global _OT_CONTEXT
    _OT_CONTEXT = None
    if args.device == "cpu" or args.method != "centroid_sinkhorn":
        return {"requested_device": args.device, "ot_device": "cpu", "ot_dtype": "float64",
                "sinkhorn_solver": args.sinkhorn_solver if args.method == "centroid_sinkhorn" else None}
    import torch
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Install CUDA-enabled PyTorch or pass --device cpu")
    torch.cuda.set_device(device)
    dtype = {"float64": torch.float64, "float32": torch.float32}[args.ot_dtype]
    # Fail before loading vectors if this GPU/build cannot execute CUDA kernels.
    probe = torch.eye(2, device=device, dtype=dtype)
    (probe @ probe).sum().item()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    _OT_CONTEXT = (torch, device, dtype, args.sinkhorn_solver)
    name = torch.cuda.get_device_name(device)
    print(f"[OT device] {device}: {name}; dtype={args.ot_dtype}; solver={args.sinkhorn_solver}", flush=True)
    return {"requested_device": args.device, "ot_device": str(device),
            "ot_dtype": args.ot_dtype, "sinkhorn_solver": args.sinkhorn_solver,
            "gpu_name": name, "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda}


def torch_ground_cost_matrix(av, bv, ground_cost, torch):
    if av.ndim != 2 or bv.ndim != 2 or av.shape[1] != bv.shape[1]:
        raise ValueError("Embedding matrices must be 2-D with matching dimensions")
    if ground_cost not in GROUND_COSTS:
        raise ValueError(f"Invalid ground cost: {ground_cost}")
    if ground_cost in ("normalized_euclidean", "cosine"):
        an = torch.linalg.vector_norm(av, dim=1, keepdim=True)
        bn = torch.linalg.vector_norm(bv, dim=1, keepdim=True)
        if bool((an <= 0).any().item()) or bool((bn <= 0).any().item()):
            raise ValueError("Normalized costs require nonzero vectors")
        left, right = av / an, bv / bn
    else:
        left, right = av, bv
    if ground_cost == "cosine":
        return (1.0 - left @ right.T).clamp_(0.0, 2.0)
    cost = left @ right.T
    cost.mul_(-2)
    cost.add_((left * left).sum(dim=1)[:, None])
    cost.add_((right * right).sum(dim=1)[None, :])
    cost.clamp_(min=0)
    if ground_cost == "normalized_squared_euclidean":
        maximum = cost.max()
        if float(maximum.item()) > 0:
            cost.div_(maximum)
        return cost
    return cost.sqrt_()


def torch_sinkhorn_transport(a, b, reg, max_iter, tolerance, ground_cost, context):
    import ot
    torch, device, dtype, solver = context
    total_start = time.perf_counter()
    with torch.inference_mode():
        if _PROFILE_PHASES:
            torch.cuda.synchronize(device)
        stamp = time.perf_counter()
        aw, av = _gpu_distribution(a, context)
        bw, bv = _gpu_distribution(b, context)
        if _PROFILE_PHASES:
            torch.cuda.synchronize(device)
            _PROFILE["gpu_transfer_wall_seconds"] += time.perf_counter() - stamp
        stamp = time.perf_counter()
        cost = torch_ground_cost_matrix(av, bv, ground_cost, torch)
        if _PROFILE_PHASES:
            torch.cuda.synchronize(device)
            _PROFILE["gpu_cost_wall_seconds"] += time.perf_counter() - stamp
        stamp = time.perf_counter()
        plan = ot.sinkhorn(aw, bw, cost, reg, method=solver,
                           numItermax=max_iter, stopThr=tolerance, warn=True)
        error = torch.maximum((plan.sum(dim=1) - aw).abs().max(),
                              (plan.sum(dim=0) - bw).abs().max())
        residual = float(error.item())
        if not bool(torch.isfinite(plan).all().item()) or not math.isfinite(residual) or residual > tolerance * 10:
            raise RuntimeError(f"Sinkhorn did not converge: marginal residual {residual}")
        result = float((plan * cost).sum().item())
        if not math.isfinite(result):
            raise RuntimeError("Nonfinite transport distance")
        if _PROFILE_PHASES:
            _PROFILE["gpu_solve_check_wall_seconds"] += time.perf_counter() - stamp
        _PROFILE["gpu_ot_calls"] += 1
        _PROFILE["gpu_ot_wall_seconds"] += time.perf_counter() - total_start
        return result



def transport(
    a,
    b,
    method,
    reg,
    max_iter,
    tolerance,
    max_cells,
    ground_cost="euclidean",
):
    import ot

    aw, av = a
    bw, bv = b
    if len(aw) * len(bw) > max_cells:
        raise MemoryError(
            "OT cost matrix exceeds --max-cells; increase explicitly (no silent truncation)"
        )
    if method == "sinkhorn" and _OT_CONTEXT is not None:
        return torch_sinkhorn_transport(a, b, reg, max_iter, tolerance, ground_cost, _OT_CONTEXT)
    cost = ground_cost_matrix(av, bv, ground_cost)
    if method == "wmd":
        result, log = ot.emd2(aw, bw, cost, numItermax=max_iter, log=True)
        if log.get("warning"):
            raise RuntimeError(log["warning"])
    else:
        plan = ot.sinkhorn(
            aw,
            bw,
            cost,
            reg,
            # method="sinkhorn_log",
            method=getattr(transport, "cpu_solver", "sinkhorn"),
            numItermax=max_iter,
            stopThr=tolerance,
            warn=True,
        )
        error = max(
            np.max(abs(plan.sum(axis=1) - aw)), np.max(abs(plan.sum(axis=0) - bw))
        )
        if not np.isfinite(plan).all() or error > tolerance * 10:
            raise RuntimeError(f"Sinkhorn did not converge: marginal residual {error}")
        result = np.einsum(
            "ij,ij->", plan, cost
        )  # transport cost, not Sinkhorn divergence
    if not np.isfinite(result):
        raise RuntimeError("Nonfinite transport distance")
    return float(result)


def classify(
    store,
    query_sides,
    query_vectors,
    method,
    *,
    topk1=4,
    topk2=3,
    gate="gold",
    gold=None,
    weights=WEIGHTS,
    reg=0.05,
    max_iter=100000,
    tolerance=1e-7,
    max_cells=10000000,
    ground_cost="euclidean",
    ot_max_terms=None,
    query_semantic=None,
    lookahead=True,
):
    if method not in ("centroid", "centroid_wmd", "centroid_sinkhorn"):
        raise ValueError("Invalid method")
    if gate not in ("gold", "none") or topk1 < 1 or topk2 < 1:
        raise ValueError("Invalid gate/Top-K1/Top-K2")
    if method != "centroid" and topk2 > topk1:
        raise ValueError("For two-stage methods, --topk2 cannot exceed --topk1")
    if ground_cost not in GROUND_COSTS:
        raise ValueError(f"Invalid ground cost: {ground_cost}")
    if gate == "gold" and (not gold or any(str(s) not in gold for s in LAYERS)):
        raise ValueError(
            "Gold gate requires DL1/DL2/DL3; use --gate none for inference"
        )
    parents, trace = [None], []
    if not any(s is not None for s in query_sides):
        return {"predictions": [None] * 3, "trace": [], "stop": "empty_query"}

    ot_method = {
        "centroid_wmd": "wmd",
        "centroid_sinkhorn": "sinkhorn",
    }.get(method)
    ot_parts = {}

    def refined_node_score(node_id, coarse):
        """Score one node with the selected pipeline, releasing its bow immediately."""
        if method == "centroid":
            return coarse
        reference = store.reference_sides(node_id, ot_max_terms)
        score, valid = 0.0, True
        for s, weight, query_side in zip(LAYERS, weights, query_sides):
            if weight == 0 or query_side is None:
                continue
            ref_side = reference[s - 1]
            if ref_side is None:
                valid = False
                break
            score += weight * transport(query_side, ref_side, ot_method, reg,
                                        max_iter, tolerance, max_cells, ground_cost)
        if not valid:
            return math.inf
        centroid_similarity = max(0.0, min(1.0, 1.0 - coarse))
        ot_similarity = 1.0 / (1.0 + score)
        combined_similarity = centroid_similarity * ot_similarity
        ot_parts[node_id] = {
            "centroid_distance": coarse,
            "centroid_similarity": centroid_similarity,
            f"{ot_method}_distance": score,
            f"{ot_method}_similarity": ot_similarity,
            "combined_similarity": combined_similarity,
            "combined_distance": 1.0 - combined_similarity,
        }
        return 1.0 - combined_similarity

    predictions, reason = [None] * 3, "completed"
    for level in LAYERS:
        ot_parts.clear()
        start = time.perf_counter()
        heap, count, all_coarse = [], 0, []
        # Centroid-only has no K1 stage. Two-stage methods retain centroid Top-K1.
        coarse_limit = None if method == "centroid" else topk1
        # Beam expansion: pool the direct children of every previous Top-K2 node.
        # Node IDs are globally unique, but seen_ids also makes the union explicit.
        seen_ids = set()
        for candidate_parent in parents:
            for node_id, name, vector in store.candidates(
                level, candidate_parent, include_centroid=True
            ):
                if node_id in seen_ids:
                    del vector
                    continue
                seen_ids.add(node_id)
                distance = cosine_distance(query_vectors, vector, weights)
                entry = (-distance, -node_id, name)
                if coarse_limit is None:
                    all_coarse.append((distance, node_id, name))
                elif len(heap) < coarse_limit:
                    heapq.heappush(heap, entry)
                elif entry > heap[0]:
                    heapq.heapreplace(heap, entry)
                count += 1
                del vector
        selected = (
            sorted(all_coarse)
            if coarse_limit is None
            else sorted([(-d, -idx, name) for d, idx, name in heap])
        )
        all_coarse.clear()
        heap.clear()
        store.release()
        coarse_seconds = time.perf_counter() - start
        if not selected:
            reason = "no_children"
            break
        refine_start = time.perf_counter()
        scores = []
        for coarse, node_id, name in selected:
            own_score = refined_node_score(node_id, coarse)
            child_scores = []
            if lookahead and level < LAYERS[-1] and math.isfinite(own_score):
                # Old AG look-ahead: add the mean score of every existing direct
                # child. Children are scored and released one at a time.
                for child_id, _, child_vector in store.candidates(
                    level + 1, name, include_centroid=True
                ):
                    child_coarse = cosine_distance(
                        query_vectors, child_vector, weights
                    )
                    del child_vector
                    child_score = refined_node_score(child_id, child_coarse)
                    if math.isfinite(child_score):
                        child_scores.append(child_score)
                store.release()

            # score = (own_score + \sum child_score) / (1 + child_cnt)
            child_count = len(child_scores)
            child_score_sum = float(np.sum(child_scores)) if child_scores else 0.0
            child_mean_score = (
                child_score_sum / child_count
                if child_count > 0
                else 0.0
            )
            score = (
                (own_score + child_score_sum) / (1 + child_count)
                if math.isfinite(own_score)
                else math.inf
            )
            scores.append(
                (
                    score,
                    node_id,
                    name,
                    own_score,
                    child_mean_score,
                    child_count,
                )
            )
            
            # child_mean_score = float(np.mean(child_scores)) if child_scores else 0.0
            # score = (
            #     own_score + child_mean_score if math.isfinite(own_score) else math.inf
            # )
            # scores.append(
            #     (score, node_id, name, own_score, child_mean_score, len(child_scores))
            # )


        ranked = sorted(scores)
        final_topk = ranked[:topk2]
        score, _, winner, _, _, _ = final_topk[0]
        row = {
            "level": level,
            "input_parents": parents,
            "beam_width": len(parents),
            "candidate_count": count,
            "lookahead_enabled": lookahead,
            "topk1": [n for _, _, n in selected] if method != "centroid" else None,
            "topk2": [n for _, _, n, _, _, _ in final_topk],
            "topk2_scores": {
                n: d if math.isfinite(d) else None
                for d, _, n, _, _, _ in final_topk
            },
            "retrieval_distances": {n: d for d, _, n in selected},
            "scores": {
                n: d if math.isfinite(d) else None for d, _, n, _, _, _ in scores
            },
            "own_scores": {
                n: d if math.isfinite(d) else None for _, _, n, d, _, _ in scores
            },
            "child_mean_scores": {n: d for _, _, n, _, d, _ in scores},
            "lookahead_child_counts": {n: c for _, _, n, _, _, c in scores},
            "coarse_seconds": coarse_seconds,
            "refine_seconds": time.perf_counter() - refine_start,
        }
        if ot_method is not None:
            row[f"{ot_method}_components"] = {
                n: ot_parts.get(node_id) for _, node_id, n in selected
            }
        selected.clear()
        scores.clear()
        ranked.clear()
        store.release()
        if not math.isfinite(score):
            row["winner"] = None
            trace.append(row)
            reason = "no_valid_candidate"
            break
        predictions[level - 1] = winner
        topk2_names = [n for _, _, n, _, _, _ in final_topk]
        topk2_correct = gold.get(str(level)) in topk2_names if gold else None
        row.update(
            winner=winner,
            correct=(winner == gold.get(str(level))) if gold else None,
            topk2_correct=topk2_correct,
        )
        trace.append(row)
        if gate == "gold" and not row["topk2_correct"]:
            reason = "gold_not_in_topk2"
            break
        parents = topk2_names
    return {"predictions": predictions, "trace": trace, "stop": reason}


class MemorySampler:
    def __enter__(self):
        try:
            import psutil

            process = psutil.Process()
            self.rss = lambda: process.memory_info().rss
        except ImportError:
            if sys.platform != "win32":
                raise RuntimeError("Install psutil for RSS measurement")
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
                    (n, ctypes.c_size_t)
                    for n in (
                        "peak",
                        "working",
                        "p1",
                        "p2",
                        "p3",
                        "p4",
                        "page",
                        "peakpage",
                    )
                ]

            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.GetCurrentProcess.restype = wintypes.HANDLE
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(Counters),
                wintypes.DWORD,
            ]

            def rss():
                counters = Counters()
                counters.cb = ctypes.sizeof(counters)
                if not psapi.GetProcessMemoryInfo(
                    kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb
                ):
                    raise ctypes.WinError(ctypes.get_last_error())
                return counters.working

            self.rss = rss
        self.baseline = self.peak = self.rss()
        self.event = threading.Event()

        def sample():
            while not self.event.wait(0.01):
                self.peak = max(self.peak, self.rss())

        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.peak = max(self.peak, self.rss())
        self.event.set()
        self.thread.join()


def worker(args):
    global _PROFILE_PHASES, _GPU_CACHE_BYTES, _GPU_CACHE_LIMIT
    _PROFILE.clear()
    _GPU_CACHE.clear()
    _GPU_CACHE_BYTES = 0
    _PROFILE_PHASES = args.profile_ot
    _GPU_CACHE_LIMIT = int(args.gpu_cache_mb * 1024 * 1024)
    if args.method in ("centroid_sinkhorn", "centroid_wmd"):
        __import__("ot")  # exclude lazy solver import from classification timing
    started = time.perf_counter()
    device_metadata = configure_ot_device(args)
    transport.cpu_solver = args.sinkhorn_solver
    embeddings = Embeddings(args.training, args.vectors, args.vector_format)
    store = Store(args.store)
    store.embeddings = embeddings
    store.distribution_cache_limit = int(args.reference_cache_mb * 1024 * 1024)
    setup_seconds = time.perf_counter() - started
    total, labeled, full = 0, 0, 0
    entered, correct, topk2_hits = [0] * 3, [0] * 3, [0] * 3
    classification_seconds = query_seconds = 0.0
    classification_cpu_seconds = 0.0
    query_weighting = None
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    try:
        with MemorySampler() as memory, open(
            args.queries, encoding="utf-8"
        ) as source, (output / f"{args.method}_paths.jsonl").open(
            "x", encoding="utf-8"
        ) as dest:
            for line in source:
                if not line.strip():
                    continue
                if args.limit is not None and total >= args.limit:
                    break
                profile_before = dict(_PROFILE)
                record = json.loads(line)
                record_weighting = record.get("weighting", "legacy_tf")
                if query_weighting is None:
                    query_weighting = record_weighting
                elif query_weighting != record_weighting:
                    raise ValueError("Query file mixes different weighting schemes")
                gold = {str(k): label(v) for k, v in record.get("gold", {}).items()}
                if gold and set(gold) != {"1", "2", "3"}:
                    raise ValueError(
                        "Provide all three gold labels or omit gold entirely"
                    )
                start = time.perf_counter()
                complete_query = sides(record["semantic"], embeddings)
                vectors = centroids(complete_query, embeddings.dim)
                query = (
                    complete_query
                    if args.method == "centroid"
                    else sides(record["semantic"], embeddings, args.ot_max_terms)
                )
                query_seconds += time.perf_counter() - start
                start = time.perf_counter()
                cpu_start = time.process_time()
                result = classify(
                    store,
                    query,
                    vectors,
                    args.method,
                    topk1=args.topk1,
                    topk2=args.topk2,
                    gate=args.gate,
                    gold=gold,
                    weights=args.weights,
                    reg=args.reg,
                    max_iter=args.max_iter,
                    tolerance=args.tolerance,
                    max_cells=args.max_cells,
                    ground_cost=args.ground_cost,
                    ot_max_terms=args.ot_max_terms,
                    query_semantic=record["semantic"],
                    lookahead=args.lookahead,
                )
                # elapsed = time.perf_counter() - start
                # classification_seconds += elapsed
                cpu_elapsed = time.process_time() - cpu_start
                elapsed = time.perf_counter() - start

                classification_cpu_seconds += cpu_elapsed
                classification_seconds += elapsed
                query_profile = {k: v - profile_before.get(k, 0.0) for k, v in _PROFILE.items()}
                result.update(
                    profile=query_profile,
                    id=record.get("id", total),
                    gold=gold,
                    seconds=elapsed,
                    query_weighting=record_weighting,
                    cpu_seconds=cpu_elapsed,
                )
                dest.write(
                    json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n"
                )
                total += 1
                if gold:
                    labeled += 1
                    full += all(
                        result["predictions"][s - 1] == gold[str(s)] for s in LAYERS
                    )
                    for row in result["trace"]:
                        index = row["level"] - 1
                        entered[index] += 1
                        correct[index] += bool(row.get("correct"))
                        topk2_hits[index] += gold[str(row["level"])] in row["topk2"]
                if not args.quiet_progress:
                    prediction = result["predictions"][2]
                    actual = gold.get("3") if gold else None
                    def shown(value):
                        return str(value).zfill(3) if value is not None else "未到达/无标签"
                    status = ("正确" if prediction == actual else "错误") if actual is not None else "无标签"
                    print(f"[{total}] 最终DL3 {shown(actual)} → {shown(prediction)} {status} | 分类用时 {elapsed:.3f}秒 | CPU {cpu_elapsed:.3f}秒", flush=True)
                    if args.profile_ot:
                        print(
                            "    分段: 节点分布构建={:.3f}s; GPU传输/缓存={:.3f}s; 成本矩阵={:.3f}s; Sinkhorn及检查={:.3f}s; OT调用={:.0f}".format(
                                query_profile.get("reference_build_wall_seconds", 0),
                                query_profile.get("gpu_transfer_wall_seconds", 0),
                                query_profile.get("gpu_cost_wall_seconds", 0),
                                query_profile.get("gpu_solve_check_wall_seconds", 0),
                                query_profile.get("gpu_ot_calls", 0)), flush=True)
                del query, complete_query, vectors, result, record
    finally:
        store.close()
    if _OT_CONTEXT is not None:
        torch, device, _, _ = _OT_CONTEXT
        device_metadata["gpu_peak_allocated_mb"] = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        device_metadata["gpu_peak_reserved_mb"] = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    summary = {
        "method": args.method,
        **device_metadata,
        "profile": dict(_PROFILE),
        "reference_cache_mb": args.reference_cache_mb,
        "gpu_cache_mb": args.gpu_cache_mb,
        "profile_ot": args.profile_ot,
        "query_limit": args.limit,
        "host_cpu_timing_note": "CPU process time excludes GPU compute time; wall time waits for each GPU OT result",
        "retrieval": "centroid",
        "second_stage": {
            "centroid": None,
            "centroid_wmd": "wmd",
            "centroid_sinkhorn": "sinkhorn",
        }[args.method],
        "score_fusion": (
            "centroid_similarity_x_ot_similarity"
            if args.method != "centroid"
            else "centroid_distance_only"
        ),
        "documents": total,
        "labeled_documents": labeled,
        "query_weighting": query_weighting,
        "topk1": args.topk1 if args.method != "centroid" else None,
        "topk2": args.topk2,
        "gate": args.gate,
        "lookahead": args.lookahead,
        "weights": args.weights,
        "reg": args.reg,
        "ground_cost": (
            args.ground_cost
            if args.method in ("centroid_sinkhorn", "centroid_wmd")
            else None
        ),
        "ot_max_terms": args.ot_max_terms,
        "ot_embedding_weighting": (
            (
                "unit_embedding_with_feature_weight_marginal"
                if args.ground_cost in ("normalized_euclidean", "cosine")
                else "original_embedding_with_feature_weight_marginal"
            )
            if args.method in ("centroid_sinkhorn", "centroid_wmd")
            else None
        ),
        "ot_cost_normalization": (
            "per_matrix_max"
            if args.method in ("centroid_sinkhorn", "centroid_wmd")
            and args.ground_cost == "normalized_squared_euclidean"
            else None
        ),
        "max_cells": args.max_cells,
        "setup_seconds": setup_seconds,
        "query_seconds": query_seconds,
        "classification_seconds": classification_seconds,
        "classification_cpu_seconds": classification_cpu_seconds,
        "classification_cpu_seconds_per_query": (
            classification_cpu_seconds / total if total else None
        ),
        "classification_wall_seconds_per_query": (
            classification_seconds / total if total else None
        ),
        "pipeline_seconds": query_seconds + classification_seconds,
        "baseline_rss_mb": memory.baseline / 2**20,
        "sampled_peak_rss_mb": memory.peak / 2**20,
        "full_path_accuracy": full / labeled if labeled else None,
    }
    for s in LAYERS:
        summary.update(
            {
                f"DL{s}_entered": entered[s - 1],
                f"DL{s}_overall_accuracy": (
                    correct[s - 1] / labeled if labeled else None
                ),
                f"DL{s}_conditional_accuracy": (
                    correct[s - 1] / entered[s - 1] if entered[s - 1] else None
                ),
                f"DL{s}_topk2_hits": topk2_hits[s - 1],
                f"DL{s}_topk2_recall": (
                    topk2_hits[s - 1] / entered[s - 1]
                    if entered[s - 1]
                    else None
                ),
            }
        )
    (output / f"{args.method}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    commands = ap.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    for arg in ("refs", "hierarchy", "store"):
        prep.add_argument("--" + arg, required=True)
    source = prep.add_mutually_exclusive_group(required=True)
    source.add_argument("--categories")
    source.add_argument("--bows")
    qp = commands.add_parser("queries")
    for arg in ("training", "input", "output"):
        qp.add_argument("--" + arg, required=True)
    qp.add_argument(
        "--idf-categories",
        required=True,
        help=(
            "Category root containing level1/, or the level1 directory itself; "
            "all DL1 DF statistics are merged for global Query TF-IDF"
        ),
    )
    qp.add_argument("--text-column", default="clean_description")
    qp.add_argument("--limit", type=int)
    for command in ("compare", "worker"):
        run = commands.add_parser(command)
        for arg in ("store", "queries", "output"):
            run.add_argument("--" + arg, required=True)
        run.add_argument("--training", required=True)
        run.add_argument("--vectors", required=True)
        run.add_argument("--vector-format", default="fasttext_vec")
        run.add_argument(
            "--topk1",
            type=int,
            default=4,
            help="Centroid candidates retained for WMD/Sinkhorn refinement",
        )
        run.add_argument(
            "--topk2",
            type=int,
            default=3,
            help="Final ranked candidates returned at each DDC level",
        )
        run.add_argument("--gate", choices=["gold", "none"], default="gold")
        run.add_argument(
            "--lookahead",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Use child-mean look-ahead (default: enabled); use --no-lookahead to disable",
        )
        run.add_argument("--weights", type=float, nargs=3, default=list(WEIGHTS))
        run.add_argument("--device", default="cuda:0", help="cuda:0, cuda:1, or cpu; only Sinkhorn OT is GPU-enabled")
        run.add_argument("--ot-dtype", choices=["float64", "float32"], default="float64")
        run.add_argument("--sinkhorn-solver", choices=["sinkhorn", "sinkhorn_log"], default="sinkhorn")
        run.add_argument("--quiet-progress", action="store_true")
        run.add_argument("--limit", type=int, help="Only process the first N queries")
        run.add_argument("--reference-cache-mb", type=float, default=512)
        run.add_argument("--gpu-cache-mb", type=float, default=256)
        run.add_argument("--profile-ot", action="store_true", help="Synchronize and time GPU phases; adds measurement overhead")
        run.add_argument("--reg", type=float, default=0.05)
        run.add_argument(
            "--ground-cost",
            choices=GROUND_COSTS,
            default="euclidean",
            help=(
                "OT ground cost (default: euclidean). "
                "normalized_squared_euclidean divides each cost matrix by its maximum"
            ),
        )
        run.add_argument("--max-iter", type=int, default=100000)
        run.add_argument("--tolerance", type=float, default=1e-7)
        run.add_argument("--max-cells", type=int, default=200000000)
        run.add_argument(
            "--ot-max-terms",
            type=int,
            help="Keep this many highest-weight terms per semantic layer for WMD/Sinkhorn",
        )
        if command == "worker":
            run.add_argument(
                "--method",
                choices=["centroid", "centroid_wmd", "centroid_sinkhorn"],
                required=True,
            )
    args = ap.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "queries":
        queries(args)
    else:
        if args.limit is not None and args.limit < 1:
            ap.error("--limit must be positive")
        if any(not math.isfinite(v) or v < 0 for v in (args.reference_cache_mb, args.gpu_cache_mb)):
            ap.error("Cache limits must be finite and nonnegative")
        weight_sum = sum(args.weights)
        if (
            args.topk1 < 1
            or args.topk2 < 1
            or args.reg <= 0
            or args.max_iter < 1
            or args.max_cells < 1
            or (args.ot_max_terms is not None and args.ot_max_terms < 1)
            or args.tolerance <= 0
            or any(not math.isfinite(w) or w < 0 for w in args.weights)
            or not math.isfinite(weight_sum)
            or weight_sum <= 0
        ):
            ap.error(
                "K/reg/iterations/tolerance/cells must be positive; weights must be finite, nonnegative, and not all zero"
            )
        # Treat layer weights as relative proportions. This accepts convenient
        # rounded inputs such as 0.333, 0.333, 0.333 and records normalized values.
        args.weights = [w / weight_sum for w in args.weights]
        if args.command == "worker":
            if args.method != "centroid" and args.topk2 > args.topk1:
                ap.error("For two-stage methods, --topk2 cannot exceed --topk1")
            worker(args)
        else:
            if not args.vectors or not args.training:
                ap.error("--training and --vectors are required for compare")
            # Separate fresh processes prevent cross-method caches and RSS contamination.
            output = Path(args.output)
            output.mkdir(parents=True, exist_ok=True)
            if any(output.glob("*_paths.jsonl")):
                raise FileExistsError(
                    "Use a fresh result directory for each experiment"
                )
            rows = []
            for method in ("centroid", "centroid_wmd", "centroid_sinkhorn"):
                subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "worker",
                        *sys.argv[2:],
                        "--method",
                        method,
                    ],
                    check=True,
                )
                rows.append(
                    json.loads(
                        (output / f"{method}_summary.json").read_text(encoding="utf-8")
                    )
                )
            with (output / "comparison.csv").open(
                "w", newline="", encoding="utf-8-sig"
            ) as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)


if __name__ == "__main__":
    main()
