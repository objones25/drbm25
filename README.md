# drbm25

PPO training of an LLM query rewriter with a BM25 reward (CS279 DeepRetrieval reproduction).

## Setup

```bash
java -version          # Java 21 is required by pyserini 2.x
uv sync                # creates .venv from uv.lock (Python 3.12)
uv sync --extra vllm   # Linux only: adds vllm==0.19.0
uv run pytest          # default run deselects gpu and integration tests
uv run ruff check .
```
