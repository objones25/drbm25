"""Publish a run's selected policy, with its provenance and results, to the Hugging Face Hub, verified.

    python -m drbm25.publish --run-dir DIR [--eval-jsonl F] [--report F] [--namespace N]
        [--repo-prefix P] [--private | --public] [--code-url URL] [--dry-run]

One private model repo per run, ``<namespace>/<prefix>-<model slug>-<reward mode>`` (namespace: the
``HF_TOKEN`` owner from ``whoami`` unless ``--namespace``; slug ``qwen2.5-3b`` for a full fine-tune,
``qwen2.5-1.5b-lora`` for LoRA). Repo contents:

  full layout   ``model.safetensors`` (best/'s actor, HF names) + the base model's ``config.json``,
                ``generation_config.json`` and tokenizer files: ``AutoModelForCausalLM.from_pretrained(repo)``.
  lora layout   ``adapter_model.safetensors`` + ``adapter_config.json`` (base model id and revision) + the
                tokenizer files: ``PeftModel.from_pretrained(base, repo)``.
  both          ``README.md`` (model card), the base model's ``LICENSE`` (and ``NOTICE`` for the Qwen research
                license, which requires it), ``provenance/best_meta.json``, ``provenance/run_manifest.json``,
                ``provenance/trained-<mode>.jsonl`` (the test-set eval), ``provenance/report.md``.

Resume checkpoints (optimizer state) are never uploaded. Only files whose hash differs from the Hub copy
are committed, so a re-run uploads nothing new. After the commit the repo tree is read back and every file
is compared with the local one: LFS files by sha256 and size, regular files by git blob id and size. Only
when all match is ``UPLOADED.json`` committed to the repo and written to ``<run_dir>/UPLOADED.json``. When
both runs of the volume (``normal`` and ``count_once`` beside ``run_dir``) have a current marker, the CLI
logs "safe to delete the volume".

huggingface_hub 0.36.2 facts (``inspect.signature`` / ``getsource`` on the installed package):
  * ``whoami()`` GETs ``/api/whoami-v2``; the token's role is ``["auth"]["accessToken"]["role"]``
    (``read``, ``write`` or ``fineGrained``; ``get_token_permission`` is deprecated in favour of this).
  * ``create_repo(repo_id, *, private, exist_ok)``; ``create_commit(repo_id, operations, *, commit_message,
    parent_commit)`` with ``CommitOperationAdd(path_in_repo, path_or_fileobj)``; ``model_info(repo_id).sha``.
  * ``list_repo_tree(repo_id, *, recursive, revision)`` yields ``RepoFile(path, size, blob_id, lfs)``; ``lfs``
    is ``BlobLfsInfo(size, sha256, pointer_size)`` for LFS files and None for regular ones, whose
    ``blob_id`` is the git object id ``sha1(b"blob <size>\\0" + content)``.
  * Errors are ``requests`` exceptions (``HfHubHTTPError`` subclasses ``requests.HTTPError``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Protocol

import requests
import yaml
from huggingface_hub import CommitOperationAdd, CommitOperationDelete
from huggingface_hub.errors import HfHubHTTPError
from huggingface_hub.hf_api import RepoFile, RepoFolder
from peft import LoraConfig
from safetensors.torch import load_file, save_file

from drbm25.checkpoint import BEST_DIR, BestLayout, BestMeta, read_best_meta
from drbm25.eval.records import read_header, read_records
from drbm25.manifest import MANIFEST_FILE, read_manifest
from drbm25.obs import configure

logger = logging.getLogger(__name__)

UPLOADED_FILE: Final = "UPLOADED.json"
DEFAULT_REPO_PREFIX: Final = "drbm25"
DEFAULT_CODE_URL: Final = "https://github.com/objones25/drbm25"  # scripts/runpod/launch.sh DEFAULT_REPO_URL
_RUN_MODES: Final = ("normal", "count_once")
_PROVENANCE: Final = "provenance"
_BEST_META_IN_REPO: Final = f"{_PROVENANCE}/best_meta.json"
_ATTEMPTS: Final = 4
_BACKOFF_S: Final = (2.0, 5.0, 15.0)
_CHUNK_BYTES: Final = 1 << 20
# Qwen RESEARCH LICENSE AGREEMENT section 3c: copies must carry this notice in a "Notice" text file.
_QWEN_RESEARCH: Final = "qwen-research"
_QWEN_RESEARCH_NOTICE: Final = (
    "Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c) Alibaba Cloud. All Rights Reserved.\n"
)
_TOKENIZER_FILES: Final = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
_FULL_CONFIG_FILES: Final = ("config.json", "generation_config.json")
_LORA_ACTOR: Final = re.compile(r"^(?P<module>.+)\.(?P<ab>lora_[AB])\.actor\.weight$")
_PEFT_PREFIX: Final = "base_model.model."

class TokenPermissionError(RuntimeError):
    """``HF_TOKEN`` cannot create and write model repos in the namespace."""


class VerificationError(RuntimeError):
    """The Hub copy of a file is missing or differs from the local file."""


# ---------------------------------------------------------------------------- Hub API surface
class _HasSha(Protocol):
    @property
    def sha(self) -> str | None: ...


class HubApi(Protocol):
    """The ``huggingface_hub.HfApi`` methods this module calls (tests inject a fake)."""

    def whoami(self) -> dict[str, Any]: ...

    def create_repo(self, repo_id: str, *, private: bool | None = ..., exist_ok: bool = ...) -> object: ...

    def model_info(self, repo_id: str) -> _HasSha: ...

    def list_repo_tree(
        self, repo_id: str, *, recursive: bool = ..., revision: str | None = ...
    ) -> Iterable[RepoFile | RepoFolder]: ...

    def create_commit(
        self,
        repo_id: str,
        operations: Iterable[CommitOperationAdd | CommitOperationDelete],
        *,
        commit_message: str,
        parent_commit: str | None = ...,
    ) -> object: ...


@dataclass(frozen=True)
class BaseFiles:
    """The base model's card text (for its license metadata) and the files copied into the repo."""

    card: str
    files: dict[str, Path]  # path in repo -> local file


BaseFetcher = Callable[[str, str, BestLayout, Path], BaseFiles]


# ---------------------------------------------------------------------------- run contents
@dataclass(frozen=True)
class RunFiles:
    """Everything published for one run, read and cross-checked by :func:`load_run`."""

    run_dir: Path
    meta: BestMeta
    manifest: dict[str, Any]
    eval_jsonl: Path
    report: Path
    n_test_queries: int
    test_ndcg_normal: float
    test_ndcg_count_once: float
    trained_steps: int | None

    @property
    def model_name(self) -> str:
        return str(self.manifest["model"]["name"])

    @property
    def model_revision(self) -> str:
        return str(self.manifest["model"]["revision"])


def model_slug(model_name: str, train_mode: str) -> str:
    """``Qwen/Qwen2.5-3B-Instruct`` + full -> ``qwen2.5-3b``; + lora -> ``qwen2.5-3b-lora``."""
    slug = model_name.rsplit("/", 1)[-1].lower().removesuffix("-instruct")
    return f"{slug}-lora" if train_mode == "lora" else slug


def _report_row(report_text: str, source: str) -> tuple[str, str]:
    for line in report_text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 3 and cells[0] == source:
            return cells[1], cells[2]
    raise ValueError(f"the report has no row for source {source!r}")


def load_run(run_dir: Path, *, eval_jsonl: Path | None = None, report: Path | None = None) -> RunFiles:
    """Read ``run_dir``'s best/ meta, manifest, test-set eval and report; ValueError when they disagree.

    Defaults (the RunPod and Expanse layout): ``<run_dir>/../eval/trained-<reward_mode>.jsonl`` and
    ``<run_dir>/../eval/report.md``. The eval file must score this best/ (same step), and the report's row
    for the run must show the eval file's means at the report's 4 decimals.
    """
    meta = read_best_meta(run_dir / BEST_DIR)
    manifest = read_manifest(run_dir)
    if manifest is None:
        raise FileNotFoundError(f"{run_dir} has no {MANIFEST_FILE}")
    if manifest.get("reward_mode") != meta.reward_mode:
        raise ValueError(f"{run_dir}: manifest reward_mode {manifest.get('reward_mode')!r} != best/ {meta.reward_mode!r}")
    source = f"trained-{meta.reward_mode}"
    eval_jsonl = eval_jsonl or run_dir.parent / "eval" / f"{source}.jsonl"
    report = report or run_dir.parent / "eval" / "report.md"
    header = read_header(eval_jsonl) or {}
    best = header.get("best") or {}
    if best.get("step") != meta.step:
        raise ValueError(f"{eval_jsonl} scores best step {best.get('step')!r}, but {run_dir}/best is step {meta.step}")
    records = read_records(eval_jsonl)
    if not records or {r.source for r in records} != {source}:
        raise ValueError(f"{eval_jsonl} must hold records of source {source!r} only")
    normal = sum(r.ndcg_normal for r in records) / len(records)
    count_once = sum(r.ndcg_count_once for r in records) / len(records)
    row = _report_row(report.read_text(encoding="utf-8"), source)
    if row != (f"{normal:.4f}", f"{count_once:.4f}"):
        raise ValueError(f"{report} shows {row} for {source}, but {eval_jsonl} averages {normal:.4f}, {count_once:.4f}")
    return RunFiles(
        run_dir=run_dir,
        meta=meta,
        manifest=manifest,
        eval_jsonl=eval_jsonl,
        report=report,
        n_test_queries=len(records),
        test_ndcg_normal=normal,
        test_ndcg_count_once=count_once,
        trained_steps=header.get("trained_steps"),
    )


# ---------------------------------------------------------------------------- model card
def license_metadata(base_card: str) -> dict[str, str]:
    """``license`` (+ ``license_name``, ``license_link``) from a model card's YAML front matter."""
    match = re.match(r"^---\n(.*?)\n---", base_card, flags=re.DOTALL)
    front = yaml.safe_load(match.group(1)) if match else None
    if not isinstance(front, dict) or not front.get("license"):
        raise ValueError("the base model card has no license in its YAML front matter")
    return {k: str(front[k]) for k in ("license", "license_name", "license_link") if front.get(k)}


def build_model_card(run: RunFiles, base_card: str, *, repo_id: str, code_url: str) -> str:
    """README.md for the repo; deterministic (no timestamps) so an unchanged run re-uploads nothing."""
    meta, lic = run.meta, license_metadata(base_card)
    full = meta.layout == "full"
    total_steps = run.manifest["config"]["run"]["total_steps"]
    git = "unknown" if meta.git_sha is None else f"[{meta.git_sha}]({code_url}/tree/{meta.git_sha})"
    if meta.git_dirty:
        git += " (with uncommitted changes)"
    front = "".join(f"{k}: {v}\n" for k, v in lic.items())
    load = (
        f'model = AutoModelForCausalLM.from_pretrained("{repo_id}")\n'
        f'tokenizer = AutoTokenizer.from_pretrained("{repo_id}")'
        if full
        else f'base = AutoModelForCausalLM.from_pretrained("{run.model_name}", revision="{run.model_revision}")\n'
        f'model = PeftModel.from_pretrained(base, "{repo_id}")\n'
        f'tokenizer = AutoTokenizer.from_pretrained("{repo_id}")'
    )
    notice = " The `NOTICE` file carries the attribution the license requires." if lic.get("license_name") == _QWEN_RESEARCH else ""
    return f"""---
{front}base_model: {run.model_name}
library_name: {"transformers" if full else "peft"}
pipeline_tag: text-generation
tags:
- drbm25
- ppo
- query-rewriting
- bm25
---

# {repo_id.rsplit("/", 1)[-1]}

Built with Qwen. A query-rewriting policy trained with PPO against a BM25 retrieval reward
(`{meta.reward_mode}` scoring), {"full fine-tune" if full else "LoRA adapter"} of `{run.model_name}`.
This is the step selected on the validation split (`{meta.selection_metric}`).

| split | nDCG@10 normal | nDCG@10 count_once |
|---|---|---|
| validation (best step {meta.step}) | {meta.val_ndcg_normal:.4f} | {meta.val_ndcg_count_once:.4f} |
| test ({run.n_test_queries} queries) | {run.test_ndcg_normal:.4f} | {run.test_ndcg_count_once:.4f} |

## Training

- reward mode: `{meta.reward_mode}`
- best step: {meta.step}
- trained steps: {run.trained_steps if run.trained_steps is not None else "unknown"} (planned {total_steps})
- base model: `{run.model_name}@{run.model_revision}`
- config sha256: `{meta.config_sha256}`
- seeds: `{json.dumps(meta.seeds, sort_keys=True)}`
- code: {code_url} at {git}

`provenance/` holds the best/ metadata, the run manifest (resolved config, versions, hardware), the test-set
evaluation (one JSON line per query) and the report comparing both runs.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
{"" if full else "from peft import PeftModel"}
{load}
```

A private repo needs a token with read access (`HF_TOKEN`).

## License

These weights are a modified version (PPO fine-tuning) of `{run.model_name}` and are distributed under that
model's license: `{lic["license"]}`{f" ({lic['license_name']})" if "license_name" in lic else ""}{f", {lic['license_link']}" if "license_link" in lic else ""}.
A copy of the license is in `LICENSE`.{notice}
"""


# ---------------------------------------------------------------------------- staging
def _lora_adapter_files(run: RunFiles, staging: Path) -> dict[str, Path]:
    """best/'s actor adapter as a PEFT adapter folder (``PeftModel.from_pretrained`` names)."""
    lora = run.manifest["config"].get("lora")
    if lora is None:
        raise ValueError(f"{run.run_dir}: a lora best/ but the manifest config has no lora section")
    renamed = {}
    for key, tensor in load_file(str(run.run_dir / BEST_DIR / "model.safetensors"), device="cpu").items():
        match = _LORA_ACTOR.match(key)
        if match is None:
            raise ValueError(f"{key!r} in {run.run_dir}/best is not an actor LoRA weight (<module>.lora_A|B.actor.weight)")
        renamed[f"{_PEFT_PREFIX}{match['module']}.{match['ab']}.weight"] = tensor
    save_file(renamed, str(staging / "adapter_model.safetensors"), metadata={"format": "pt"})
    LoraConfig(
        r=lora["rank"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        target_modules=list(lora["target_modules"]),
        bias="none",
        task_type="CAUSAL_LM",
        base_model_name_or_path=run.model_name,
        revision=run.model_revision,
    ).save_pretrained(str(staging))
    return {"adapter_model.safetensors": staging / "adapter_model.safetensors", "adapter_config.json": staging / "adapter_config.json"}


def stage_files(
    run: RunFiles, base_fetcher: BaseFetcher, staging: Path, *, repo_id: str, code_url: str
) -> dict[str, Path]:
    """The upload list, path in repo -> local file; generated files are written into ``staging``."""
    staging.mkdir(parents=True, exist_ok=True)
    base = base_fetcher(run.model_name, run.model_revision, run.meta.layout, staging / "base")
    plan: dict[str, Path] = dict(base.files)
    if run.meta.layout == "full":
        plan["model.safetensors"] = run.run_dir / BEST_DIR / "model.safetensors"
    else:
        plan.update(_lora_adapter_files(run, staging))
    card = staging / "README.md"
    card.write_text(build_model_card(run, base.card, repo_id=repo_id, code_url=code_url), encoding="utf-8")
    plan["README.md"] = card
    if license_metadata(base.card).get("license_name") == _QWEN_RESEARCH:
        notice = staging / "NOTICE"
        notice.write_text(_QWEN_RESEARCH_NOTICE, encoding="utf-8")
        plan["NOTICE"] = notice
    plan[_BEST_META_IN_REPO] = run.run_dir / BEST_DIR / "meta.json"
    plan[f"{_PROVENANCE}/{MANIFEST_FILE}"] = run.run_dir / MANIFEST_FILE
    plan[f"{_PROVENANCE}/{run.eval_jsonl.name}"] = run.eval_jsonl
    plan[f"{_PROVENANCE}/report.md"] = run.report
    if "LICENSE" not in plan:
        raise ValueError(f"the base model files of {run.model_name} have no LICENSE")
    return plan


def hub_base_fetcher(name: str, revision: str, layout: BestLayout, dest: Path) -> BaseFiles:
    """The base model's README (license metadata), LICENSE, tokenizer and (full) config files at ``revision``.

    Uses the local HF cache when the snapshot is there (run_all.sh and prepare_login.sh download it); a
    missing file raises ``EntryNotFoundError``.
    """
    from huggingface_hub import hf_hub_download  # network code: imported on the CLI path only

    del dest  # cached files are uploaded from the HF cache as they are

    def get(filename: str) -> Path:
        return Path(_with_retries(lambda: hf_hub_download(name, filename, revision=revision), sleep=time.sleep))

    names = ("LICENSE", *_TOKENIZER_FILES, *(_FULL_CONFIG_FILES if layout == "full" else ()))
    card = get("README.md").read_text(encoding="utf-8")
    return BaseFiles(card=card, files={n: get(n) for n in names})


# ---------------------------------------------------------------------------- hashes and verification
@dataclass(frozen=True)
class FileDigest:
    size: int
    sha256: str
    git_blob_sha1: str  # the id the Hub reports for a regular (non-LFS) file


def digest(path: Path) -> FileDigest:
    size = path.stat().st_size
    sha256 = hashlib.sha256()
    blob = hashlib.sha1(f"blob {size}\0".encode())  # noqa: S324 -- git's object id, compared with the Hub's
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK_BYTES), b""):
            sha256.update(chunk)
            blob.update(chunk)
    return FileDigest(size=size, sha256=sha256.hexdigest(), git_blob_sha1=blob.hexdigest())


def _mismatch(local: FileDigest, remote: RepoFile | None) -> str | None:
    if remote is None:
        return "missing on the Hub"
    if remote.lfs is not None:
        if (remote.lfs.sha256, remote.lfs.size) != (local.sha256, local.size):
            return f"LFS sha256/size {remote.lfs.sha256}/{remote.lfs.size} != local {local.sha256}/{local.size}"
        return None
    if (remote.blob_id, remote.size) != (local.git_blob_sha1, local.size):
        return f"git blob {remote.blob_id}/{remote.size} != local {local.git_blob_sha1}/{local.size}"
    return None


def _remote_files(tree: Iterable[RepoFile | RepoFolder]) -> dict[str, RepoFile]:
    return {f.path: f for f in tree if isinstance(f, RepoFile)}


def files_to_upload(local: Mapping[str, FileDigest], tree: Iterable[RepoFile | RepoFolder]) -> list[str]:
    remote = _remote_files(tree)
    return sorted(path for path, d in local.items() if _mismatch(d, remote.get(path)) is not None)


def verify_remote(local: Mapping[str, FileDigest], tree: Iterable[RepoFile | RepoFolder]) -> None:
    """Raise VerificationError naming every file whose Hub copy is missing or differs."""
    remote = _remote_files(tree)
    problems = [f"{path}: {why}" for path, d in sorted(local.items()) if (why := _mismatch(d, remote.get(path)))]
    if problems:
        raise VerificationError("Hub copy does not match: " + "; ".join(problems))


# ---------------------------------------------------------------------------- token
def check_write_access(whoami: Mapping[str, Any]) -> str:
    """The token owner's name when the token can create and write their model repos; else TokenPermissionError.

    Roles (huggingface_hub 0.36.2 ``get_token_permission``): ``write`` passes, ``read`` fails. A
    ``fineGrained`` token passes when one of its ``scoped`` entries names the user and grants ``repo.write``
    (the whoami-v2 shape: secondary sources, not in the library; the Hub's own 403 on create_repo is the
    backstop). Anything else (OAuth, no token) fails.
    """
    name = whoami.get("name")
    token = (whoami.get("auth") or {}).get("accessToken") or {}
    role = token.get("role")
    if not isinstance(name, str) or not name:
        raise TokenPermissionError("HF_TOKEN: whoami returned no user name")
    if role == "write":
        return name
    if role == "read":
        raise TokenPermissionError("HF_TOKEN is read-only: create a token with the 'write' role at https://huggingface.co/settings/tokens")
    if role == "fineGrained":
        scoped = (token.get("fineGrained") or {}).get("scoped") or []
        if any((s.get("entity") or {}).get("name") == name and "repo.write" in (s.get("permissions") or []) for s in scoped):
            return name
        raise TokenPermissionError(f"HF_TOKEN is a fine-grained token without repo.write on user {name!r}; grant it or use a 'write' token")
    raise TokenPermissionError(f"HF_TOKEN: cannot tell whether this token may write (role {role!r}); use a 'write' token")


# ---------------------------------------------------------------------------- retries
def _is_network_error(exc: Exception) -> bool:
    if isinstance(exc, requests.ConnectionError | requests.Timeout):
        return True
    if isinstance(exc, HfHubHTTPError) and exc.response is not None:
        return exc.response.status_code >= 500 or exc.response.status_code == 429
    return False


def _with_retries[T](call: Callable[[], T], *, sleep: Callable[[float], None]) -> T:
    """``call()`` with up to ``_ATTEMPTS`` tries on network errors (connection, timeout, HTTP 5xx/429)."""
    for attempt in range(1, _ATTEMPTS + 1):
        try:
            return call()
        except (requests.RequestException, HfHubHTTPError) as exc:
            if attempt == _ATTEMPTS or not _is_network_error(exc):
                raise
            wait_s = _BACKOFF_S[attempt - 1]
            logger.warning("Hub call failed (%s); retry %d/%d in %.0fs", type(exc).__name__, attempt, _ATTEMPTS - 1, wait_s, exc_info=True)
            sleep(wait_s)
    raise AssertionError("unreachable: the last attempt returns or raises")


# ---------------------------------------------------------------------------- publish
@dataclass(frozen=True)
class PublishResult:
    repo_id: str
    revision: str
    uploaded: list[str]  # paths committed by this call (empty when the Hub already had everything)


def _head(api: HubApi, repo_id: str, sleep: Callable[[float], None]) -> str:
    sha = _with_retries(lambda: api.model_info(repo_id), sleep=sleep).sha
    if not sha:
        raise RuntimeError(f"the Hub returned no head commit for {repo_id}")
    return sha


def _tree(api: HubApi, repo_id: str, revision: str, sleep: Callable[[float], None]) -> list[RepoFile | RepoFolder]:
    return _with_retries(lambda: list(api.list_repo_tree(repo_id, recursive=True, revision=revision)), sleep=sleep)


def _commit_missing(
    api: HubApi,
    repo_id: str,
    files: Mapping[str, Path | bytes],
    digests: Mapping[str, FileDigest],
    *,
    message: str,
    sleep: Callable[[float], None],
    delete_marker: bool = False,
) -> tuple[str, list[str]]:
    """Commit the files whose Hub copy differs, re-reading the tree on every attempt, so a retry after a
    commit whose response was lost uploads nothing twice. Returns (head after, paths committed).

    ``delete_marker``: the same commit removes the repo's UPLOADED.json, so the Hub never shows a
    verification marker next to files that have not been verified yet.
    """
    committed: list[str] = []
    for attempt in range(1, _ATTEMPTS + 1):
        head = _head(api, repo_id, sleep)
        tree = _tree(api, repo_id, head, sleep)
        missing = files_to_upload(digests, tree)
        if not missing:
            return head, committed
        ops: list[CommitOperationAdd | CommitOperationDelete] = [
            CommitOperationAdd(path_in_repo=p, path_or_fileobj=files[p]) for p in missing
        ]
        if delete_marker and UPLOADED_FILE in _remote_files(tree):
            ops.append(CommitOperationDelete(path_in_repo=UPLOADED_FILE))
        try:
            api.create_commit(repo_id, ops, commit_message=message, parent_commit=head)
        except (requests.RequestException, HfHubHTTPError) as exc:
            if attempt == _ATTEMPTS or not _is_network_error(exc):
                raise
            logger.warning("commit to %s failed (%s); retry %d/%d", repo_id, type(exc).__name__, attempt, _ATTEMPTS - 1, exc_info=True)
            sleep(_BACKOFF_S[attempt - 1])
            continue
        committed = sorted({*committed, *missing})
        logger.info("committed %d file(s) to %s: %s", len(missing), repo_id, ", ".join(missing))
    return _head(api, repo_id, sleep), committed


def _bytes_digest(data: bytes) -> FileDigest:
    blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data)  # noqa: S324 -- git's object id
    return FileDigest(size=len(data), sha256=hashlib.sha256(data).hexdigest(), git_blob_sha1=blob.hexdigest())


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    tmp = path.with_name(f".tmp-{path.name}-{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def publish_run(
    api: HubApi,
    run: RunFiles,
    *,
    base_fetcher: BaseFetcher,
    namespace: str | None,
    repo_prefix: str,
    private: bool,
    code_url: str,
    dry_run: bool = False,
    sleep: Callable[[float], None] = time.sleep,
) -> PublishResult | None:
    """Upload ``run`` and verify it; write ``UPLOADED.json`` only after verification. None on ``dry_run``.

    The token's write access is checked before anything is created or uploaded.
    """
    user = check_write_access(_with_retries(api.whoami, sleep=sleep))
    slug = model_slug(run.model_name, run.meta.layout)
    repo_id = f"{namespace or user}/{repo_prefix}-{slug}-{run.meta.reward_mode}"
    with tempfile.TemporaryDirectory(prefix="drbm25-publish-") as tmp:
        plan = stage_files(run, base_fetcher, Path(tmp), repo_id=repo_id, code_url=code_url)
        digests = {path: digest(local) for path, local in plan.items()}
        if dry_run:
            for path in sorted(plan):
                logger.info("dry run: would upload %s (%d bytes) from %s to %s", path, digests[path].size, plan[path], repo_id)
            return None
        _with_retries(lambda: api.create_repo(repo_id, private=private, exist_ok=True), sleep=sleep)
        git = run.meta.git_sha or "unknown"
        message = f"drbm25 {run.meta.reward_mode}: best step {run.meta.step} (code {git})"
        revision, uploaded = _commit_missing(
            api, repo_id, dict(plan), digests, message=message, sleep=sleep, delete_marker=True
        )
        verify_remote(digests, _tree(api, repo_id, revision, sleep))
    files = {path: {"sha256": d.sha256, "size": d.size} for path, d in sorted(digests.items())}
    marker = {
        "repo_id": repo_id,
        "revision": revision,
        "reward_mode": run.meta.reward_mode,
        "best_step": run.meta.step,
        "files": files,
        "verified_at": datetime.now(UTC).isoformat(),
    }
    _publish_remote_marker(api, marker, sleep=sleep)
    _write_json_atomic(run.run_dir / UPLOADED_FILE, marker)
    logger.info("verified %d files in %s at %s; wrote %s", len(files), repo_id, revision, run.run_dir / UPLOADED_FILE)
    return PublishResult(repo_id=repo_id, revision=revision, uploaded=uploaded)


def _publish_remote_marker(api: HubApi, marker: Mapping[str, Any], *, sleep: Callable[[float], None]) -> None:
    """Commit UPLOADED.json to the repo (``launch.sh --status`` reads it once the pod is gone) and verify it.

    The Hub copy leaves out ``revision`` and ``verified_at`` so that it is byte-identical on a re-run of an
    unchanged run, which then commits nothing; it is committed on top of the verified revision.
    """
    repo_id = str(marker["repo_id"])
    remote = {k: marker[k] for k in ("repo_id", "reward_mode", "best_step", "files")}
    data = (json.dumps(remote, indent=2, sort_keys=True) + "\n").encode()
    digests = {UPLOADED_FILE: _bytes_digest(data)}
    message = f"drbm25: verified {len(marker['files'])} files at {marker['revision']}"
    revision, _ = _commit_missing(api, repo_id, {UPLOADED_FILE: data}, digests, message=message, sleep=sleep)
    verify_remote(digests, _tree(api, repo_id, revision, sleep))


# ---------------------------------------------------------------------------- volume
def run_uploaded(run_dir: Path) -> bool:
    """True when ``run_dir/UPLOADED.json`` exists and was written for the current best/ (same meta.json)."""
    marker_path = run_dir / UPLOADED_FILE
    meta_path = run_dir / BEST_DIR / "meta.json"
    if not marker_path.is_file() or not meta_path.is_file():
        return False
    recorded = json.loads(marker_path.read_text(encoding="utf-8"))["files"].get(_BEST_META_IN_REPO)
    return recorded is not None and recorded["sha256"] == digest(meta_path).sha256


def volume_safe(runs_dir: Path) -> bool:
    """True when both runs (``normal``, ``count_once``) under ``runs_dir`` are uploaded and current."""
    return all(run_uploaded(runs_dir / mode) for mode in _RUN_MODES)


# ---------------------------------------------------------------------------- CLI
def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m drbm25.publish", description="Publish a run's best policy to the Hugging Face Hub, verified.")
    parser.add_argument("--run-dir", type=Path, required=True, help="a run directory holding best/ and run_manifest.json")
    parser.add_argument("--eval-jsonl", type=Path, default=None, help="default: RUN_DIR/../eval/trained-<mode>.jsonl")
    parser.add_argument("--report", type=Path, default=None, help="default: RUN_DIR/../eval/report.md")
    parser.add_argument("--namespace", default=None, help="user or organisation (default: the HF_TOKEN owner)")
    parser.add_argument("--repo-prefix", default=DEFAULT_REPO_PREFIX)
    visibility = parser.add_mutually_exclusive_group()
    visibility.add_argument("--private", dest="private", action="store_true", default=True, help="(default)")
    visibility.add_argument("--public", dest="private", action="store_false")
    parser.add_argument("--code-url", default=DEFAULT_CODE_URL, help="https URL of the code repository")
    parser.add_argument("--dry-run", action="store_true", help="list what would be uploaded; no Hub writes")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Exit 0 published and verified (or dry run), 1 any failure (no UPLOADED.json is written)."""
    from huggingface_hub import HfApi  # network code: imported on the CLI path only

    ns = _parse_args(argv)
    configure(level="INFO", fmt="json")
    run = load_run(ns.run_dir, eval_jsonl=ns.eval_jsonl, report=ns.report)
    # HfApi reads HF_TOKEN from the environment; the token is never passed around or logged here.
    api: HubApi = HfApi()
    result = publish_run(
        api,
        run,
        base_fetcher=hub_base_fetcher,
        namespace=ns.namespace,
        repo_prefix=ns.repo_prefix,
        private=ns.private,
        code_url=ns.code_url,
        dry_run=ns.dry_run,
    )
    if result is not None:
        logger.info("published %s at %s (%d file(s) uploaded now)", result.repo_id, result.revision, len(result.uploaded))
        if volume_safe(ns.run_dir.parent):
            logger.info("both runs are uploaded and verified: safe to delete the volume")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
