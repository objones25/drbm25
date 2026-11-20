"""Logging and W&B plumbing (vendored from the observability-expert skill's obslog).

Entry points call `configure(...)` once; library code only uses
`logging.getLogger(__name__)`. W&B pieces live in `drbm25.obs.runs` and
`drbm25.obs.wandb_run` and are imported on demand, so this package does not
import wandb.
"""

from drbm25.obs.setup import configure

__all__ = ["configure"]
