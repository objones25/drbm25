# Results

Every result below ends with a **Reproducibility** line: the seeds, the config sha256 and the git sha that
produced it. For trained rows these come from the run's `best/meta.json` (step, validation scores, config
sha256, seeds) and `run_manifest.json` (git sha and dirty flag, every seed and its derivation), and from the
eval jsonl headers; the bootstrap seed is printed in `report.md`. The test set for every row is all 2,305
dev queries; validation (checkpoint selection) uses 500 held-out train queries, never dev.

## Exp 1 baseline: original queries (no rewrite), 2026-11-20

BM25 (k1 = 0.9, b = 0.4) over the 800K-document MS MARCO-Science collection, nDCG@10, all 2,305
MS-S dev queries, scored by both retrieval modes. Original queries only: no model is involved.

| source | nDCG@10 normal | nDCG@10 count_once |
|---|---|---|
| original | 0.3877 | 0.3877 |

Unrounded: normal 0.3877261, count_once 0.3877292. The modes differ on 30 of 2,305 queries, because
only 5.2% of original queries repeat an analyzed term; 971 queries score 0 in normal mode, 970 in count_once, 965 in both; no query hit
Lucene's clause limit.

Query statistics (analyzed terms, original queries): mean terms 5.22, mean unique terms 5.15,
repeat_ratio 0.012, any_repeat_rate 0.052, format_error_rate 0.000.

- Reproducibility: no seed affects this row (no model, no sampling, no train/validation split: the
  test set is all of dev) and no bootstrap was run (one source). The run predates the reproducibility
  manifest, so its exact git sha and config sha256 were not recorded; the code is that of commit `e15264e`
  (2026-11-20), which added this file. The scoring path for original queries has not changed since, so
  the 0.3877 baseline stays valid.
- Index: `indexes/msmarco_800k`, 800,000 docs, checksum `c8c8c528be768b57fcf9e23e6b8ccbb7842e7a4dbeb0ac9e5de12900044e495e`.
- Dataset revision: `44eb79992e8c5ed4784dbe6559f8b097d7d03855` (`DeepRetrieval/datasets`).
- Command (Java 21 on PATH; per-query records in `runs/eval/original.jsonl`, not committed):

```
python -m drbm25.eval.run_eval --config configs/runpod_a100_full.yaml --source original \
    --out runs/eval/original.jsonl \
    --set retrieval.index_dir=indexes/msmarco_800k --set data.data_dir=data/msmarco
python -m drbm25.eval.report --inputs runs/eval/original.jsonl --out runs/eval/report_original.md
```

Check: the normal-mode nDCG@10 was recomputed for every one of the 2,305 queries with plain pyserini
(`LuceneSearcher.search`, `set_bm25(0.9, 0.4)`) and matched the recorded values to 1e-9. By hand on
single queries: dev-7 ("3 characteristics of protozoa replication") has its relevant passage at rank 6,
nDCG = 1/log2(7) = 0.3562; dev-8 ("what is protoplasm") has it at rank 4, nDCG = 1/log2(5) = 0.4307;
dev-4 ("what is probability biology") does not retrieve it in the top 10, nDCG = 0.
