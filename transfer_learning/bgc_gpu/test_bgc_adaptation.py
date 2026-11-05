"""Small CPU checks; no real BGC training or CUDA device is required."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

import bgc_workflow as flow
import pipelines_bgc_gpu as core


class AdaptationTests(unittest.TestCase):
    def test_streamed_tfidf_matches_original_builder(self):
        builder = flow.load_module(flow.HERE / 'feature_builder_node.py', 'original_builder')
        texts = ['The model uses a model for book classification.',
                 'The book describes a library and an author.',
                 'An author designs a computer model for science.']
        expected, expected_df, count = builder.aggregate_category_features(texts)
        actual = flow.aggregate_streaming(texts, builder, 2, 1e-6)
        self.assertEqual(actual['total_docs'], count)
        self.assertEqual(actual['df_counts'], expected_df)
        for s in (1, 2, 3):
            self.assertEqual(set(actual['category_features'][s]), set(expected[s]))
            for term in expected[s]:
                self.assertAlmostEqual(actual['category_features'][s][term], expected[s][term], places=12)

    def test_full_cpu_smoke(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / 'data'
            data.mkdir()
            paths = [('Fiction', 'Fantasy', 'Epic Fantasy'),
                     ('Nonfiction', 'Business', 'Marketing')]
            texts = ['The dragon guards a castle and a sword.',
                     'The wizard explores a forest and a kingdom.',
                     'The company sells a product to a customer.',
                     'The manager plans a campaign for a brand.']
            nodes = [{'path': list(p[:i]), 'level': i}
                     for p in paths for i in (1, 2, 3)]
            (data / 'hierarchy.json').write_text(json.dumps({'nodes': nodes}), encoding='utf-8')
            rows = [{'id': str(i), 'description': text,
                     **dict(zip(('DL1', 'DL2', 'DL3'), paths[i // 2]))}
                    for i, text in enumerate(texts)]
            for split in ('train', 'dev'):
                (data / f'{split}.jsonl').write_text(
                    ''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
            words = sorted(set('dragon castle sword wizard forest kingdom company product '
                               'customer manager campaign brand guard explore sell plan'.split()))
            rng = np.random.default_rng(42)
            vector_file = root / 'tiny.vec'
            vector_file.write_text(f'{len(words)} 4\n' + ''.join(
                word + ' ' + ' '.join(map(str, rng.normal(size=4))) + '\n'
                for word in words), encoding='utf-8')
            def run(script, *args):
                result = subprocess.run([sys.executable, str(flow.HERE / script), *map(str, args)],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            model, store = root / 'model', root / 'nodes.sqlite'
            run('bgc_workflow.py', 'train', '--data', data, '--model', model)
            # Completed nodes must be safely reusable when the inputs are unchanged.
            run('bgc_workflow.py', 'train', '--data', data, '--model', model)
            run('bgc_workflow.py', 'prepare', '--model', model, '--store', store,
                '--vectors', vector_file)
            queries, distributions = root / 'queries.jsonl', root / 'ot.sqlite'
            run('bgc_workflow.py', 'queries', '--model', model, '--input', data / 'dev.jsonl',
                '--output', queries)
            run('prepare_ot_distributions.py', '--pipeline', flow.HERE / 'pipelines_bgc_gpu.py',
                '--store', store, '--training', flow.HERE, '--vectors', vector_file,
                '--ot-max-terms', '10', '--output', distributions)
            for method in ('centroid', 'centroid_sinkhorn'):
                output = root / method
                run('pipelines_bgc_gpu.py', 'worker', '--store', store, '--queries', queries,
                    '--output', output, '--training', flow.HERE, '--vectors', vector_file,
                    '--method', method, '--device', 'cpu', '--precomputed-ot', distributions,
                    '--ot-max-terms', '10', '--ground-cost', 'cosine', '--reg', '0.5',
                    '--topk1-per-level', '2', '2', '2', '--topk2-per-level', '1', '2', '2',
                    '--weights', '0.33', '0.33', '0.33', '--no-lookahead')
                summary = json.loads((output / f'{method}_summary.json').read_text())
                self.assertEqual(summary['documents'], 4)
                self.assertEqual(summary['topk2'], [1, 2, 2])
                records = [json.loads(x) for x in (output / f'{method}_paths.jsonl').read_text().splitlines()]
                for r in records:
                    self.assertIn(r['predictions'][0], ('Fiction', 'Nonfiction'))
                    self.assertEqual(len(r['trace'][0]['topk2']), 1)
                    self.assertEqual(r['trace'][0]['requested_topk2'], 1)


if __name__ == '__main__':
    unittest.main()
