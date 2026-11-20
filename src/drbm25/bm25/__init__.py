"""BM25 index building and the normal / count-once searcher.

Nothing is re-exported here on purpose: importing pyserini starts a JVM, so
`drbm25.bm25.index.tsv_to_jsonl` stays importable without Java.
"""
