# BGC / large_model GPU adapter

This is an independent copy. It does not edit the original DDC files.

## Layout on Windows

```text
transfer_learning/
  bgc_prepared/                 # Existing deduplicated data
    train.jsonl
    dev.jsonl
    test.jsonl
    hierarchy.json
  bgc_gpu/                     # Unzip this package here
    bgc_workflow.py
    pipelines_bgc_gpu.py
    ...
```

Keep the FastText .vec file wherever it already exists. Pass its path with
`--vectors` or `-Vectors`; copying the multi-GB file is unnecessary.
Relative paths in run_bgc_dev.ps1 are interpreted from the script directory.

## Environment

Activate the existing CUDA-enabled virtual environment used for DDC. Verify:

```powershell
python -c "import numpy, spacy, ot, torch; print(torch.cuda.is_available()); print(spacy.load('en_core_web_sm').meta['name'])"
```

If the spaCy model alone is missing:

```powershell
python -m spacy download en_core_web_sm
```

Dependencies: numpy, spacy, en_core_web_sm, POT, CUDA-enabled torch. The retained
legacy prepare/queries CSV/XLSX entry points additionally need pandas/openpyxl;
the BGC adapters do not use those entry points. Do not replace a working CUDA
torch installation merely to run the data adapters.

## One script: train -> store -> Dev queries -> Centroid -> OT -> GPU Sinkhorn

From bgc_gpu, replace the example vector path with a valid relative path:

```powershell
powershell -ExecutionPolicy Bypass -File ./run_bgc_dev.ps1 -Vectors ../wiki-news-300d-1M-subword.vec
```

Defaults: all Dev queries, terms=1000, reg=0.05, cosine, ordinary Sinkhorn,
float64, cuda:0, equal semantic weights, Lookahead, gold gate.

DL1/DL2/DL3 K1 = 3/8/8; K2 = 2/4/4. These are an initial experiment setup, not
parameters selected on Dev. K1=3 retains all three DL1 categories. K2=2 permits
measurable DL1 gating loss. Candidate lists automatically contain fewer than K
when fewer nodes exist. Only one DL2 child exists under the selected Children's
Books branch; pooling children of the entire preceding beam still follows the
original pipeline. Keep these semantics in mind when interpreting recall.

To retain all DL1 categories, pass `-K2 3,4,4` when invoking the script from an
already running PowerShell session (use `& ./run_bgc_dev.ps1 ...`).

The script stops on an error. Existing databases, Query files, OT distributions,
and result files are not overwritten. Run an individual `-Stage` after fixing
an error; completed training nodes can be resumed by the Train stage.
Available stages: Train, Prepare, Queries, Centroid, OTPrepare, Sinkhorn.

Example: after completing training and database construction, first generate
and classify 10 Dev records to check execution (not to estimate accuracy):

```powershell
& ./run_bgc_dev.ps1 -Vectors ../wiki-news-300d-1M-subword.vec -Stage Queries -Limit 10
& ./run_bgc_dev.ps1 -Vectors ../wiki-news-300d-1M-subword.vec -Stage Centroid -Limit 10
& ./run_bgc_dev.ps1 -Vectors ../wiki-news-300d-1M-subword.vec -Stage OTPrepare
& ./run_bgc_dev.ps1 -Vectors ../wiki-news-300d-1M-subword.vec -Stage Sinkhorn -Limit 10
```

## Equivalent individual commands

```powershell
python bgc_workflow.py train --data ../bgc_prepared --model ./bgc_model

python bgc_workflow.py prepare --model ./bgc_model --store ./nodes_bgc.sqlite --vectors ../wiki-news-300d-1M-subword.vec

python bgc_workflow.py queries --model ./bgc_model --input ../bgc_prepared/dev.jsonl --output ./queries_bgc_dev_all.jsonl

python pipelines_bgc_gpu.py worker `
  --store ./nodes_bgc.sqlite `
  --queries ./queries_bgc_dev_all.jsonl `
  --output ./result_bgc_dev_centroid_manual `
  --training . `
  --vectors ../wiki-news-300d-1M-subword.vec `
  --method centroid `
  --device cpu `
  --topk1-per-level 3 8 8 `
  --topk2-per-level 2 4 4 `
  --lookahead --gate gold --weights 0.33 0.33 0.33

python prepare_ot_distributions.py `
  --pipeline ./pipelines_bgc_gpu.py `
  --store ./nodes_bgc.sqlite `
  --training . `
  --vectors ../wiki-news-300d-1M-subword.vec `
  --ot-max-terms 1000 `
  --output ./ot_bgc_terms1000.sqlite

python pipelines_bgc_gpu.py worker `
  --store ./nodes_bgc.sqlite `
  --queries ./queries_bgc_dev_all.jsonl `
  --precomputed-ot ./ot_bgc_terms1000.sqlite `
  --output ./result_bgc_dev_sinkhorn_manual `
  --training . `
  --vectors ../wiki-news-300d-1M-subword.vec `
  --method centroid_sinkhorn `
  --device cuda:0 --ot-dtype float64 --sinkhorn-solver sinkhorn `
  --ground-cost cosine --reg 0.05 --ot-max-terms 1000 --max-cells 1000000 `
  --topk1-per-level 3 8 8 `
  --topk2-per-level 2 4 4 `
  --lookahead --gate gold --weights 0.33 0.33 0.33 --profile-ot
```

For WMD, use `--method centroid_wmd --device cpu` and a fresh output directory;
the same OT distributions can be reused. WMD does not use entropy regularization.

## Training and representation

- Only train.jsonl builds reference features and global Query IDF.
- Each training blurb contributes once to each of its three ancestor nodes.
  All eligible Train documents are used; there is no independent per-level sampling.
- Features reuse the supplied large_model extractor: original description,
  spaCy en_core_web_sm, no set deduplication before pairs/triples, document-level
  set deduplication for DF, node-local log((N+1)/(df+1)) IDF, and normalized
  feature weights below 1e-6 removed. All three semantic layers are used at
  every classification level.
- Training streams one node at a time. It accumulates sum(TF) and DF, then
  multiplies by node IDF; this is algebraically equivalent to summing per-doc
  TF*IDF. Very small floating-point accumulation differences are possible.
- Original category names and explicit parent relations are stored in SQLite;
  filenames use safe nXXXX identifiers, including for names containing colons.
- Reference centroids preserve the old weighted-sum embedding formula and are
  not truncated to the OT term limit. OT term limits apply to transport supports.
- categories/level1 contains disjoint training-document partitions, so merging
  their DF counts yields the global Query IDF without counting ancestors twice.
- Data file hashes, extractor hash, and spaCy model version are saved in the
  training manifest. Changed training inputs require a new model directory;
  dependent stores/distributions/queries must also be regenerated.
- Nodes containing a single training document have zero node-local IDF for all
  observed terms. Empty semantic layers or OOV vectors can cause invalid OT
  candidates under the inherited scoring policy; these are not silently filled
  from Dev or Test. Review warnings before large evaluations.

## Dev/Test and metrics

The runner uses Dev only. After choosing a configuration, create Test queries:

```powershell
python bgc_workflow.py queries --model ./bgc_model --input ../bgc_prepared/test.jsonl --output ./queries_bgc_test.jsonl
```

Use the same frozen database and selected worker arguments, changing only
`--queries` and `--output`. Do not tune parameters using Test.

The summary and per-query JSONL retain the original field names. BGC `name`
values are strings rather than DDC numbers. Summary topk1/topk2 become arrays
when per-level overrides are used; each trace also records requested_topk1 and
requested_topk2. There is no DDC prefix inference.

Gold gating is an evaluation mode: a Query proceeds if its true label remains
in the preceding retained beam, even when Top-1 is wrong. Use `--gate none` for
label-independent inference. For a report, distinguish these modes rather than
calling gold-gated timings unconditional deployment latency.

DL1 K2=2 means only Top-2 is retained there, so it cannot provide a stored
three-item output. Do not label that retained list as a complete DL1 Top-3.
Classification wall time includes online distribution loading, GPU transfer,
and OT work; text feature extraction, Query embeddings, setup, and offline OT
preparation are separately recorded. Profiling adds synchronization overhead.

## Validation

```powershell
python -m unittest test_bgc_adaptation -v
```

The tests compare streamed TF-IDF against the original builder and run a small
CPU end-to-end fixture, including string labels, explicit parents, resumable
training, Query generation, OT precomputation, Centroid, Sinkhorn, and per-level K.
They do not benchmark BGC accuracy or verify CUDA execution on the target GPU.
