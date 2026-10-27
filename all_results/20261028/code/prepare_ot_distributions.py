#!/usr/bin/env python3
"""Persist reference-node OT distributions once; no test data or GPU needed."""
import argparse
import importlib.util
import json
import os
import pickle
import sqlite3
import time
from pathlib import Path


def fingerprint(path):
    stat = Path(path).stat()
    return {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pipeline', default='./pipelines_gpu_precomputed.py')
    ap.add_argument('--store', required=True)
    ap.add_argument('--training', default='.')
    ap.add_argument('--vectors', default='./wiki-news-300d-1M-subword.vec')
    ap.add_argument('--vector-format', default='fasttext_vec')
    ap.add_argument('--ot-max-terms', type=int, default=1000)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    if args.ot_max_terms < 1:
        ap.error('--ot-max-terms must be positive')
    target = Path(args.output).resolve()
    temporary = target.with_name(target.name + '.partial')
    for path in (target, temporary):
        if path.exists():
            raise FileExistsError(f'Choose a new output name; file exists: {path}')
    for path in (args.pipeline, args.store, args.vectors):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    before_store, before_vectors = fingerprint(args.store), fingerprint(args.vectors)
    spec = importlib.util.spec_from_file_location('ot_distribution_pipeline', args.pipeline)
    pipeline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline)
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    embeddings = pipeline.Embeddings(args.training, args.vectors, args.vector_format)
    source = pipeline.Store(args.store)
    target.parent.mkdir(parents=True, exist_ok=True)
    destination = sqlite3.connect(temporary)
    empty_layers = 0
    try:
        destination.execute('PRAGMA journal_mode=DELETE')
        destination.execute('CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)')
        destination.execute('CREATE TABLE distributions(node_id INTEGER PRIMARY KEY,data BLOB NOT NULL)')
        nodes = source.db.execute('SELECT id,name,level FROM nodes ORDER BY id').fetchall()
        if not nodes:
            raise ValueError('No reference nodes found')
        for count, (node_id, name, level) in enumerate(nodes, 1):
            start = time.perf_counter()
            raw = source.bow(node_id)
            distributions = pipeline.sides(raw, embeddings, args.ot_max_terms)
            del raw
            empty_layers += sum(item is None for item in distributions)
            destination.execute('INSERT INTO distributions VALUES(?,?)',
                                (node_id, pickle.dumps(distributions, protocol=pickle.HIGHEST_PROTOCOL)))
            del distributions
            if count % 10 == 0:
                destination.commit()
            print(f'[{count}/{len(nodes)}] DL{level} {str(name).zfill(3)}: {time.perf_counter()-start:.3f}s', flush=True)
        if before_store != fingerprint(args.store) or before_vectors != fingerprint(args.vectors):
            raise RuntimeError('Source database or vectors changed while preparing')
        metadata = dict(format_version=1, completed=True, node_count=len(nodes),
                        ot_max_terms=args.ot_max_terms, vector_format=args.vector_format,
                        vector_dimension=embeddings.dim, distribution_dtype='float64',
                        source_store=before_store, source_vectors=before_vectors,
                        empty_semantic_layers=empty_layers,
                        preparation_wall_seconds=time.perf_counter()-wall_start,
                        preparation_cpu_seconds=time.process_time()-cpu_start)
        destination.execute('INSERT INTO metadata VALUES(?,?)', ('configuration', json.dumps(metadata)))
        destination.commit()
    finally:
        destination.close()
        source.close()
    os.replace(temporary, target)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f'Saved persistent reference distributions: {target}')
    print(f'File size: {target.stat().st_size / 1024**3:.3f} GiB')


if __name__ == '__main__':
    main()
