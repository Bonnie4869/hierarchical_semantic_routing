"""BGC data adapters for the existing large_model GPU classification core.

All learned features, document frequencies, and centroids use train.jsonl only.
Node-local TF-IDF matches feature_builder_node.py; streaming avoids retaining
all spaCy Docs and tuple sequences for an entire broad category in memory.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import pickle
import sqlite3
import time

HERE = Path(__file__).resolve().parent


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_rows(path):
    rows = []
    with Path(path).open(encoding='utf-8-sig') as source:
        for line in source:
            if line.strip():
                row = json.loads(line)
                if not row.get('description', '').strip():
                    raise ValueError(f'Empty description in {path}: {row.get("id")}')
                for level in (1, 2, 3):
                    if not isinstance(row.get(f'DL{level}'), str) or not row[f'DL{level}']:
                        raise ValueError(f'Missing string DL{level}: {row.get("id")}')
                rows.append(row)
    if not rows:
        raise ValueError(f'Empty dataset: {path}')
    return rows


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def aggregate_streaming(texts, builder, batch_size, min_weight):
    """sum_d TF(d,t)*IDF_node(t), then layer normalization and filtering."""
    df = {s: Counter() for s in (1, 2, 3)}
    tf_sums = {s: defaultdict(float) for s in (1, 2, 3)}
    count = 0
    for doc in builder.nlp.pipe(texts, batch_size=batch_size):
        doc_sets, sequences = builder._extract_doc_features(doc)
        for layer, terms, unique in zip((1, 2, 3), sequences, doc_sets):
            df[layer].update(unique)
            if terms:
                size = len(terms)
                for term, frequency in Counter(terms).items():
                    tf_sums[layer][term] += frequency / size
        count += 1
        if count % 200 == 0:
            print(f'    Parsed {count} documents', flush=True)
    features = {}
    for layer in (1, 2, 3):
        raw = {term: value * math.log((count + 1) / (df[layer][term] + 1))
               for term, value in tf_sums[layer].items()}
        total = sum(raw.values())
        features[layer] = {term: value / total for term, value in raw.items()
                           if total > 0 and value > 0 and value / total >= min_weight}
        print(f'    Semantic L{layer}: {len(raw):,} -> {len(features[layer]):,}', flush=True)
    return {'category_features': features, 'df_counts': df, 'total_docs': count}


def train(args):
    data = Path(args.data)
    rows = read_rows(data / 'train.jsonl')
    original = json.loads((data / 'hierarchy.json').read_text(encoding='utf-8'))['nodes']
    valid_paths = {tuple(n['path']) for n in original if n['level'] == 3}
    groups, parent_map, names = defaultdict(list), {}, {}
    for row in rows:
        path = tuple(row[f'DL{i}'] for i in (1, 2, 3))
        if path not in valid_paths:
            raise ValueError(f'Training path absent from hierarchy: {path}')
        for level, name in enumerate(path, 1):
            key, parent = (level, name), path[level - 2] if level > 1 else None
            if key in parent_map and parent_map[key] != parent:
                raise ValueError(f'Name reused under multiple parents: {key}')
            parent_map[key] = parent
            names[key] = path[:level]
            groups[key].append(row['description'])
    configuration = {'train_sha256': digest(data / 'train.jsonl'),
                     'hierarchy_sha256': digest(data / 'hierarchy.json'),
                     'builder_sha256': digest(HERE / 'feature_builder_node.py'),
                     'min_weight': args.min_weight,
                     'representation': 'node_local_tfidf_original_text_no_precombination_dedup',
                     'sampling': 'all_train_rows_aggregated_to_ancestors'}
    model = Path(args.model)
    model.mkdir(parents=True, exist_ok=True)
    manifest_path = model / 'manifest.json'
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text(encoding='utf-8'))
    builder = load_module(HERE / 'feature_builder_node.py', 'bgc_features')
    configuration['spacy_model'] = builder.nlp.meta.get('name')
    configuration['spacy_model_version'] = builder.nlp.meta.get('version')
    # Check the full configuration including the tokenizer model on resume.
    if manifest_path.exists():
        if prior['configuration'] != configuration:
            raise ValueError('Training inputs/configuration/spaCy model changed; choose a new --model directory')
    nodes = []
    for index, ((level, name), texts) in enumerate(sorted(groups.items())):
        relative = f'categories/level{level}/category_n{index:04d}.pkl'
        nodes.append({'id': index, 'name': name, 'level': level,
                      'parent': parent_map[(level, name)], 'path': list(names[(level, name)]),
                      'file': relative, 'documents': len(texts)})
    manifest = {'configuration': configuration, 'nodes': nodes, 'completed': False}
    atomic_json(manifest_path, manifest)
    for index, node in enumerate(nodes, 1):
        target = model / node['file']
        if target.exists():
            print(f'[{index}/{len(nodes)}] Reusing DL{node["level"]}: {node["name"]}', flush=True)
            continue
        print(f'[{index}/{len(nodes)}] DL{node["level"]}: {node["name"]}, '
              f'documents={node["documents"]}', flush=True)
        result = aggregate_streaming(groups[(node['level'], node['name'])], builder,
                                     args.batch_size, args.min_weight)
        result['node'] = node
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix('.pkl.partial')
        with temporary.open('wb') as out:
            pickle.dump(result, out, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(target)
    manifest['completed'] = True
    atomic_json(manifest_path, manifest)
    print(f'Training completed: {len(nodes)} nodes, {len(rows)} unique train documents')


def load_manifest(model):
    manifest = json.loads((model / 'manifest.json').read_text(encoding='utf-8'))
    if not manifest.get('completed'):
        raise ValueError('Training is incomplete; resume the train command first')
    return manifest


def prepare(args):
    import numpy as np
    model, target = Path(args.model), Path(args.store)
    manifest = load_manifest(model)
    partial = target.with_name(target.name + '.building')
    if target.exists() or partial.exists():
        raise FileExistsError(f'Store/partial file already exists: {target}')
    pipeline = load_module(HERE / 'pipelines_bgc_gpu.py', 'bgc_core')
    embeddings = pipeline.Embeddings(HERE, args.vectors, args.vector_format)
    target.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(partial)
    try:
        db.execute('CREATE TABLE nodes (id INTEGER PRIMARY KEY, level INTEGER, name TEXT, '
                   'parent TEXT, centroid BLOB, bow BLOB, UNIQUE(level,name))')
        db.execute('CREATE INDEX child_lookup ON nodes(level,parent)')
        for node in manifest['nodes']:
            with (model / node['file']).open('rb') as source:
                semantic = pickle.load(source)['category_features']
            # Preserve the original reference-centroid definition: weighted sum
            # of feature embeddings after filtering, without OT term truncation.
            centroid = np.zeros((3, embeddings.dim), dtype=np.float64)
            for layer in (1, 2, 3):
                for term, weight in semantic[layer].items():
                    centroid[layer - 1] += weight * embeddings.term(term)
            if not np.isfinite(centroid).all():
                raise ValueError(f'Nonfinite centroid: {node["name"]}')
            db.execute('INSERT INTO nodes VALUES (?,?,?,?,?,?)',
                       (node['id'], node['level'], node['name'], node['parent'],
                        pickle.dumps(centroid, protocol=pickle.HIGHEST_PROTOCOL),
                        pickle.dumps(semantic, protocol=pickle.HIGHEST_PROTOCOL)))
            db.commit()
            print(f'DL{node["level"]}: {node["name"]}', flush=True)
        db.execute('CREATE TABLE bgc_metadata (key TEXT PRIMARY KEY, value TEXT)')
        db.execute('INSERT INTO bgc_metadata VALUES (?,?)',
                   ('configuration', json.dumps(manifest['configuration'])))
        db.commit()
    finally:
        db.close()
    partial.replace(target)
    print(f'Stored {len(manifest["nodes"])} BGC nodes: {target}')


def queries(args):
    model, target = Path(args.model), Path(args.output)
    manifest = load_manifest(model)
    rows = read_rows(args.input)
    if args.limit is not None:
        rows = rows[:args.limit]
    paths = {tuple(n['path']) for n in manifest['nodes'] if n['level'] == 3}
    for row in rows:
        if tuple(row[f'DL{i}'] for i in (1, 2, 3)) not in paths:
            raise ValueError(f'Query path is not supported by train: {row["id"]}')
    if target.exists():
        raise FileExistsError(target)
    pipeline = load_module(HERE / 'pipelines_bgc_gpu.py', 'bgc_query_core')
    builder = load_module(HERE / 'feature_builder_node.py', 'bgc_query_features')
    global_df, total_docs, _ = pipeline.load_global_document_frequencies(model / 'categories')
    target.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with target.open('x', encoding='utf-8') as out:
        docs = builder.nlp.pipe((row['description'] for row in rows), batch_size=args.batch_size)
        for index, (row, doc) in enumerate(zip(rows, docs), 1):
            _, sequences = builder._extract_doc_features(doc)
            semantic = {s: pipeline.query_tfidf(seq, global_df[s], total_docs)
                        for s, seq in zip((1, 2, 3), sequences)}
            out.write(json.dumps({'id': row['id'], 'isbn': row.get('isbn'),
                                  'semantic': semantic,
                                  'gold': {str(i): row[f'DL{i}'] for i in (1, 2, 3)},
                                  'weighting': 'global_tfidf'}, ensure_ascii=False) + '\n')
            if index % 100 == 0 or index == len(rows):
                print(f'Prepared {index}/{len(rows)} queries', flush=True)
    info = {'documents': len(rows), 'training_documents_for_idf': total_docs,
            'text_feature_extraction_wall_seconds': time.perf_counter() - start,
            'training_configuration': manifest['configuration']}
    atomic_json(target.with_suffix('.metadata.json'), info)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    tr = commands.add_parser('train')
    tr.add_argument('--data', default='./bgc_prepared')
    tr.add_argument('--model', default='./bgc_model')
    tr.add_argument('--batch-size', type=int, default=16)
    tr.add_argument('--min-weight', type=float, default=1e-6)
    prep = commands.add_parser('prepare')
    prep.add_argument('--model', default='./bgc_model')
    prep.add_argument('--store', default='./nodes_bgc.sqlite')
    prep.add_argument('--vectors', required=True)
    prep.add_argument('--vector-format', default='fasttext_vec')
    qp = commands.add_parser('queries')
    qp.add_argument('--model', default='./bgc_model')
    qp.add_argument('--input', required=True)
    qp.add_argument('--output', required=True)
    qp.add_argument('--batch-size', type=int, default=16)
    qp.add_argument('--limit', type=int)
    args = parser.parse_args()
    if getattr(args, 'batch_size', 1) < 1:
        parser.error('--batch-size must be positive')
    if args.command == 'train' and (not math.isfinite(args.min_weight) or args.min_weight < 0):
        parser.error('--min-weight must be finite and nonnegative')
    if getattr(args, 'limit', None) is not None and args.limit < 1:
        parser.error('--limit must be positive')
    {'train': train, 'prepare': prepare, 'queries': queries}[args.command](args)


if __name__ == '__main__':
    main()
