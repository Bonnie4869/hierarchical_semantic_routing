#!/usr/bin/env python3
"""Classify every query directly against all DL3 centroid nodes.

This is a flat DL3 evaluation: DL1 and DL2 are not run and do not gate DL3.
The script reuses the query-centroid construction in the supplied pipelines.py.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import pickle
import time
from pathlib import Path

import numpy as np


def load_pipeline(path: str):
    spec = importlib.util.spec_from_file_location("centroid_pipeline", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import pipeline: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_dl3_centroids(store):
    rows = store.db.execute(
        "SELECT name, centroid FROM nodes WHERE level=3 ORDER BY name"
    ).fetchall()
    if not rows:
        raise RuntimeError("No DL3 nodes found in the SQLite store")

    names = [str(name) for name, _ in rows]
    matrices = np.stack(
        [np.asarray(pickle.loads(blob), dtype=np.float64) for _, blob in rows]
    )
    if matrices.ndim != 3 or matrices.shape[1] != 3:
        raise ValueError(
            f"Expected DL3 centroid shape (nodes, 3, dimension), got {matrices.shape}"
        )
    if not np.isfinite(matrices).all():
        raise ValueError("DL3 centroid matrix contains non-finite values")
    return names, matrices


def all_distances(query, candidates, candidate_norms, weights):
    """Weighted three-layer cosine distance for every DL3 candidate."""
    query_norms = np.linalg.norm(query, axis=1)
    denominator = candidate_norms * query_norms[None, :]
    dots = np.einsum("nld,ld->nl", candidates, query, optimize=True)
    similarities = np.divide(
        dots,
        denominator,
        out=np.zeros_like(dots),
        where=denominator > 0,
    )
    layer_distances = 1.0 - np.clip(similarities, -1.0, 1.0)
    return layer_distances @ weights


def display_ddc(value):
    """Keep DDC labels human-readable even when SQLite stores 006 as 6."""
    text = str(value)
    return text.zfill(3) if text.isdigit() else text


def main():
    parser = argparse.ArgumentParser(
        description="Flat global DL3 centroid classification with Top-K output"
    )
    parser.add_argument("--pipeline", default="./pipelines.py")
    parser.add_argument("--store", default="./nodes.sqlite")
    parser.add_argument("--queries", default="./queries_k1.jsonl")
    parser.add_argument("--training", default=".")
    parser.add_argument("--vectors", default="./wiki-news-300d-1M-subword.vec")
    parser.add_argument("--vector-format", default="fasttext_vec")
    parser.add_argument("--output", default="./result_flat_dl3_centroid_top5")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument(
        "--weights", type=float, nargs=3, default=[0.12, 0.21, 0.67]
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    if args.topk < 1:
        parser.error("--topk must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if any(not math.isfinite(x) or x < 0 for x in args.weights):
        parser.error("--weights must be finite and nonnegative")
    weight_sum = sum(args.weights)
    if weight_sum <= 0:
        parser.error("At least one weight must be positive")
    weights = np.asarray(args.weights, dtype=np.float64) / weight_sum

    pipeline = load_pipeline(args.pipeline)
    setup_start = time.perf_counter()
    embeddings = pipeline.Embeddings(
        args.training, args.vectors, args.vector_format
    )
    store = pipeline.Store(args.store)
    try:
        names, candidate_centroids = load_dl3_centroids(store)
    finally:
        store.close()
    candidate_norms = np.linalg.norm(candidate_centroids, axis=2)
    setup_seconds = time.perf_counter() - setup_start

    topk = min(args.topk, len(names))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    paths_file = output / "flat_dl3_centroid_paths.jsonl"
    summary_file = output / "flat_dl3_centroid_summary.json"

    documents = labeled = top1_hits = topk_hits = 0
    query_seconds = classification_seconds = 0.0

    with open(args.queries, encoding="utf-8") as source, paths_file.open(
        "w", encoding="utf-8"
    ) as destination:
        for line in source:
            if not line.strip():
                continue
            if args.limit is not None and documents >= args.limit:
                break

            record = json.loads(line)
            query_start = time.perf_counter()
            distributions = pipeline.sides(record["semantic"], embeddings)
            query_centroid = pipeline.centroids(distributions, embeddings.dim)
            query_seconds += time.perf_counter() - query_start

            classify_start = time.perf_counter()
            distances = all_distances(
                query_centroid,
                candidate_centroids,
                candidate_norms,
                weights,
            )
            # Stable ordering makes equal-distance results reproducible.
            order = sorted(
                range(len(names)), key=lambda index: (distances[index], names[index])
            )[:topk]
            classification_seconds += time.perf_counter() - classify_start

            ranked_names_raw = [names[index] for index in order]
            ranked_names = [display_ddc(name) for name in ranked_names_raw]
            ranked_distances = [float(distances[index]) for index in order]
            gold_value = record.get("gold", {}).get("3")
            gold_raw = pipeline.label(gold_value) if gold_value is not None else None
            gold = display_ddc(gold_raw) if gold_raw is not None else None
            top1_correct = gold_raw is not None and ranked_names_raw[0] == gold_raw
            topk_correct = gold_raw is not None and gold_raw in ranked_names_raw

            documents += 1
            if gold is not None:
                labeled += 1
                top1_hits += int(top1_correct)
                topk_hits += int(topk_correct)

            destination.write(
                json.dumps(
                    {
                        "id": record.get("id", documents - 1),
                        "gold_dl3": gold,
                        "top5": ranked_names,
                        "top5_distances": ranked_distances,
                        "top1_correct": top1_correct,
                        "top5_correct": topk_correct,
                    },
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )

            if documents % 10 == 0:
                print(f"[progress] {documents} queries")

    pipeline_seconds = query_seconds + classification_seconds
    summary = {
        "method": "flat_global_dl3_centroid",
        "documents": documents,
        "labeled_documents": labeled,
        "dl3_candidates": len(names),
        "topk": topk,
        "weights": weights.tolist(),
        "top1_hits": top1_hits,
        "top1_accuracy": top1_hits / labeled if labeled else None,
        "top5_hits": topk_hits,
        "top5_recall": topk_hits / labeled if labeled else None,
        "setup_seconds": setup_seconds,
        "query_seconds": query_seconds,
        "classification_seconds": classification_seconds,
        "pipeline_seconds": pipeline_seconds,
        "query_seconds_per_document": query_seconds / documents if documents else None,
        "classification_seconds_per_document": (
            classification_seconds / documents if documents else None
        ),
        "pipeline_seconds_per_document": (
            pipeline_seconds / documents if documents else None
        ),
    }
    summary_file.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved: {paths_file}")
    print(f"Saved: {summary_file}")


if __name__ == "__main__":
    main()
