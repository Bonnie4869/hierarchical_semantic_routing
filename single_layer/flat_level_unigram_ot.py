#!/usr/bin/env python3
"""Flat chosen-DDC-level noun-unigram OT classification; default DL1.

Reuse trusted pipeline helpers and existing SQLite/query files. Default: score
ALL nodes at --level by pure OT. Optional --topk1 enables unigram centroid retrieval.
"""
import argparse
import importlib.util
import json
import math
import pickle
import time
from pathlib import Path

import numpy as np


def load_pipeline(path):
    spec = importlib.util.spec_from_file_location('flat_unigram_pipeline', path)
    if spec is None or spec.loader is None:
        raise ValueError(f'Cannot import {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def unigram_bag(semantic):
    return semantic.get('1', semantic.get(1, {}))


def display_label(value, pipeline):
    if value is None:
        return None
    name = pipeline.label(value)
    return name.zfill(3) if name.isdigit() else name


def ot_distance(query, reference, args, pipeline):
    import ot
    aw, av = query
    bw, bv = reference
    if len(aw) * len(bw) > args.max_cells:
        raise MemoryError('Cost matrix exceeds --max-cells')
    cost = pipeline.ground_cost_matrix(av, bv, args.ground_cost)
    if args.method == 'wmd':
        value, log = ot.emd2(aw, bw, cost, numItermax=args.max_iter, log=True)
        if log.get('warning'):
            raise RuntimeError(log['warning'])
    else:
        plan = ot.sinkhorn(
            aw, bw, cost, args.reg, method=args.sinkhorn_solver,
            numItermax=args.max_iter, stopThr=args.tolerance, warn=True,
        )
        residual = max(np.max(np.abs(plan.sum(axis=1) - aw)),
                       np.max(np.abs(plan.sum(axis=0) - bw)))
        if not np.isfinite(plan).all() or residual > args.tolerance * 10:
            raise RuntimeError(f'Sinkhorn marginal residual: {residual}')
        value = np.einsum('ij,ij->', plan, cost)
    if not math.isfinite(float(value)):
        raise RuntimeError('Nonfinite OT distance')
    return float(value)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pipeline', default='./pipelines.py')
    ap.add_argument('--store', default='./nodes.sqlite')
    ap.add_argument('--queries', default='./queries_k1.jsonl')
    ap.add_argument('--training', default='.')
    ap.add_argument('--vectors', default='./wiki-news-300d-1M-subword.vec')
    ap.add_argument('--vector-format', default='fasttext_vec')
    ap.add_argument('--method', choices=['sinkhorn', 'wmd'], default='sinkhorn')
    ap.add_argument('--level', type=int, choices=[1, 2, 3], default=1)
    ap.add_argument('--output', required=True)
    ap.add_argument('--topk', type=int, default=5)
    ap.add_argument('--topk1', type=int, default=0,
                    help='0: pure OT over all nodes at --level; positive: unigram centroid shortlist')
    ap.add_argument('--ground-cost', choices=['cosine', 'euclidean',
                    'normalized_euclidean', 'normalized_squared_euclidean'], default='cosine')
    ap.add_argument('--reg', type=float, default=0.05)
    ap.add_argument('--sinkhorn-solver', choices=['sinkhorn', 'sinkhorn_log'], default='sinkhorn')
    ap.add_argument('--ot-max-terms', type=int, default=1000)
    ap.add_argument('--max-cells', type=int, default=10000000)
    ap.add_argument('--max-iter', type=int, default=100000)
    ap.add_argument('--tolerance', type=float, default=1e-7)
    ap.add_argument('--limit', type=int)
    args = ap.parse_args()
    if args.topk < 1 or args.topk1 < 0 or args.ot_max_terms < 1 or args.max_cells < 1 or args.max_iter < 1:
        ap.error('Counts must be positive; --topk1 may be zero')
    if args.limit is not None and args.limit < 1:
        ap.error('--limit must be positive')
    if not math.isfinite(args.reg) or args.reg <= 0 or not math.isfinite(args.tolerance) or args.tolerance <= 0:
        ap.error('--reg and --tolerance must be finite and positive')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    stem = f'flat_dl{args.level}_unigram_{args.method}'
    paths_file = output / f'{stem}_paths.jsonl'
    summary_file = output / f'{stem}_summary.json'
    if paths_file.exists() or summary_file.exists():
        raise FileExistsError('Choose a fresh output directory')
    if not Path(args.store).is_file():
        raise FileNotFoundError(args.store)
    if not Path(args.queries).is_file():
        raise FileNotFoundError(args.queries)

    setup_wall = time.perf_counter()
    setup_cpu = time.process_time()
    pipeline = load_pipeline(args.pipeline)
    store = pipeline.Store(args.store)
    try:
        columns = 'id,name,centroid' if args.topk1 else 'id,name'
        rows = store.db.execute(f'SELECT {columns} FROM nodes WHERE level=? ORDER BY name', (args.level,)).fetchall()
        if not rows:
            raise ValueError(f'Database has no DL{args.level} nodes')
        names = {int(row[0]): display_label(row[1], pipeline) for row in rows}
        if len(set(names.values())) != len(names):
            raise ValueError('Duplicate normalized class labels')
        candidate_vectors = None
        if args.topk1:
            candidate_vectors = np.stack([np.asarray(pickle.loads(row[2]), dtype=float)[0] for row in rows])
            norms = np.linalg.norm(candidate_vectors, axis=1)
        node_ids = [int(row[0]) for row in rows]
        del rows
        embeddings = pipeline.Embeddings(args.training, args.vectors, args.vector_format)
        setup_wall = time.perf_counter() - setup_wall
        setup_cpu = time.process_time() - setup_cpu
        count = labeled = one_hits = empty = gold_missing = retrieval_hits = 0
        hits = [0] * min(args.topk, len(node_ids), args.topk1 or len(node_ids))
        query_wall = query_cpu = classify_wall = classify_cpu = 0.0
        with open(args.queries, encoding='utf-8') as source, paths_file.open('x', encoding='utf-8') as destination:
            for line in source:
                if not line.strip():
                    continue
                if args.limit is not None and count >= args.limit:
                    break
                record = json.loads(line)
                start_w, start_c = time.perf_counter(), time.process_time()
                bag = unigram_bag(record['semantic'])
                query = pipeline.side(bag, embeddings, args.ot_max_terms)
                q_centroid = None
                if args.topk1:
                    full = pipeline.side(bag, embeddings)
                    if full is not None:
                        q_centroid = full[0] @ full[1]
                query_wall += time.perf_counter() - start_w
                query_cpu += time.process_time() - start_c
                start_w, start_c = time.perf_counter(), time.process_time()
                selected = node_ids
                if args.topk1 and q_centroid is not None:
                    denominator = norms * np.linalg.norm(q_centroid)
                    cosine = np.divide(candidate_vectors @ q_centroid, denominator,
                                       out=np.zeros_like(denominator), where=denominator > 0)
                    distances = 1 - np.clip(cosine, -1, 1)
                    indices = sorted(range(len(node_ids)), key=lambda i: (distances[i], names[node_ids[i]]))[:args.topk1]
                    selected = [node_ids[i] for i in indices]
                scores = []
                missing_nodes = 0
                if query is None:
                    empty += 1
                else:
                    for node_id in selected:
                        reference_bag = store.bow(node_id)
                        reference = pipeline.side(unigram_bag(reference_bag), embeddings, args.ot_max_terms)
                        del reference_bag
                        if reference is None:
                            missing_nodes += 1
                            continue
                        distance = ot_distance(query, reference, args, pipeline)
                        scores.append((distance, names[node_id]))
                        del reference
                    store.release()
                scores.sort(key=lambda item: (item[0], item[1]))
                ranking = scores[:len(hits)]
                elapsed_w = time.perf_counter() - start_w
                elapsed_c = time.process_time() - start_c
                classify_wall += elapsed_w
                classify_cpu += elapsed_c
                gold_record = record.get('gold', {})
                gold = display_label(gold_record.get(str(args.level), gold_record.get(args.level)), pipeline)
                ranked = [name for _, name in ranking]
                correct = bool(gold is not None and ranked and ranked[0] == gold)
                if gold is not None:
                    labeled += 1
                    gold_missing += gold not in names.values()
                    retrieval_hits += bool(query is not None and gold in {names[i] for i in selected})
                    one_hits += correct
                    for k in range(1, len(hits) + 1):
                        hits[k-1] += gold in ranked[:k]
                destination.write(json.dumps({
                    'id': record.get('id', count), 'gold_ddc': gold, 'ddc_level': args.level,
                    'method': args.method, 'semantic_layer': 1,
                    'candidate_count': len(selected) if query is not None else 0,
                    'scored_candidates': len(scores), 'empty_reference_nodes': missing_nodes,
                    'topk': ranked, 'ot_distances': [value for value, _ in ranking],
                    'top1_correct': correct, 'topk_correct': gold in ranked if gold is not None else None,
                    'empty_query': query is None, 'classification_wall_seconds': elapsed_w,
                    'classification_cpu_seconds': elapsed_c,
                }, ensure_ascii=False, allow_nan=False) + '\n')
                count += 1
                print(f'[{count}] gold={gold} top1={ranked[0] if ranked else None} correct={correct} ({elapsed_w:.3f}s)', flush=True)
        summary = {
            'method': f'flat_dl{args.level}_unigram_{args.method}', 'documents': count,
            'labeled_documents': labeled, 'ddc_level': args.level, 'candidates': len(node_ids),
            'hierarchy': False, 'lookahead': False, 'semantic_layers': [1],
            'unigram_definition': 'existing noun/proper-noun features from semantic[1]',
            'score_fusion': None, 'ranking': 'ascending pure OT transport cost',
            'retrieval': 'unigram_centroid' if args.topk1 else None,
            'topk1': args.topk1 or None, 'topk': len(hits),
            'ground_cost': args.ground_cost, 'reg': args.reg if args.method == 'sinkhorn' else None,
            'sinkhorn_solver': args.sinkhorn_solver if args.method == 'sinkhorn' else None,
            'ot_max_terms': args.ot_max_terms, 'max_iter': args.max_iter,
            'max_cells': args.max_cells, 'tolerance': args.tolerance,
            'empty_queries': empty, 'gold_absent_from_database': gold_missing,
            'retrieval_hits': retrieval_hits if args.topk1 else None,
            'retrieval_recall': retrieval_hits / labeled if args.topk1 and labeled else None,
            'top1_hits': one_hits, 'top1_accuracy': one_hits / labeled if labeled else None,
            'topk_hits': hits[-1], 'topk_recall': hits[-1] / labeled if labeled else None,
            'recall_by_k': {str(k): hits[k-1] / labeled if labeled else None for k in range(1, len(hits)+1)},
            'setup_wall_seconds': setup_wall, 'setup_cpu_seconds': setup_cpu,
            'query_wall_seconds': query_wall, 'query_cpu_seconds': query_cpu,
            'classification_wall_seconds': classify_wall, 'classification_cpu_seconds': classify_cpu,
            'classification_wall_seconds_per_query': classify_wall / count if count else None,
            'classification_cpu_seconds_per_query': classify_cpu / count if count else None,
            'query_plus_classification_wall_seconds_per_query': (query_wall+classify_wall)/count if count else None,
            'query_plus_classification_cpu_seconds_per_query': (query_cpu+classify_cpu)/count if count else None,
            'timing_note': 'Query timing starts from saved semantic features; text extraction excluded. Setup separate.',
        }
        summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        store.close()


if __name__ == '__main__':
    main()
