"""Thin wrapper: `uv run python scripts/prepare_data.py --config <yaml> [--set k=v]`."""

from drbm25.prepare import main

raise SystemExit(main())
