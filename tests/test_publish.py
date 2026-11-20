"""drbm25.publish: upload each run's best/ policy plus provenance to the Hugging Face Hub, verified.

No network: every test injects ``FakeHub`` (a local-directory stand-in for ``HfApi``) and a fake base-model
file fetcher. ``FakeHub`` reports files the way ``HfApi.list_repo_tree`` does in huggingface_hub 0.36.2:
``RepoFile`` with ``lfs`` (sha256 + size) for LFS files and only the git blob id for regular ones.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import huggingface_hub
import pytest
import requests
import torch
from huggingface_hub import CommitOperationAdd, CommitOperationDelete
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError
from huggingface_hub.hf_api import RepoFile, RepoFolder
from peft import PeftModel
from transformers import AutoModelForCausalLM

from drbm25 import publish as publish_module
from drbm25.checkpoint import BestLayout, BestMeta, save_best
from drbm25.config import LoraCfg, ScoringMode
from drbm25.eval.records import QueryRecord, write_records
from drbm25.manifest import write_manifest
from drbm25.obs.setup import shutdown
from drbm25.ppo.models import LoraPolicyModels
from drbm25.publish import (
    UPLOADED_FILE,
    BaseFiles,
    TokenPermissionError,
    VerificationError,
    build_model_card,
    check_write_access,
    invalidate_run,
    load_run,
    model_slug,
    publish_run,
    stage_files,
    volume_safe,
)
from drbm25.publish import (
    main as publish_main,
)
from drbm25.tiny import build_tiny_qwen2, build_tiny_tokenizer

_QWEN_RESEARCH_CARD = """---
license: other
license_name: qwen-research
license_link: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/LICENSE
---

# base
"""
_LICENSE_TEXT = "Qwen RESEARCH LICENSE AGREEMENT (test copy)\n"
_LORA = LoraCfg(rank=4, alpha=8, dropout=0.0, target_modules=("q_proj", "v_proj"))


# ---------------------------------------------------------------------------- fake Hub
def _git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()  # noqa: S324 -- git's object id, not security


@dataclass
class _Info:
    sha: str | None


@dataclass(frozen=True)
class _Declared:
    """What the uploading client declared for a file (the Hub's tree reports this, not the stored bytes)."""

    size: int
    sha256: str
    blob: str


def _declared(data: bytes) -> _Declared:
    return _Declared(size=len(data), sha256=hashlib.sha256(data).hexdigest(), blob=_git_blob_sha1(data))


@dataclass
class FakeHub:
    """Stores each repo under ``root/<repo_id>/`` so ``from_pretrained(root / repo_id)`` loads what was uploaded."""

    root: Path
    user: str = "tester"
    role: str = "write"
    fine_grained: dict[str, Any] | None = None
    lfs_suffixes: tuple[str, ...] = (".safetensors",)
    corrupt: set[str] = field(default_factory=set)  # paths reported with a wrong hash after upload
    corrupt_storage: set[str] = field(default_factory=set)  # stored bytes damaged; the declared hash stays right
    declared: dict[tuple[str, str], _Declared] = field(default_factory=dict)
    downloads: list[str] = field(default_factory=list)
    drop: set[str] = field(default_factory=set)  # paths silently not stored by a commit
    network_failures: int = 0  # the next N create_commit calls raise ConnectionError before doing anything
    commit_error: Exception | None = None  # every create_commit raises this
    commit_attempts: int = 0
    info_errors: list[Exception] = field(default_factory=list)  # raised, in order, by the next model_info calls
    commits: list[list[str]] = field(default_factory=list)
    commit_messages: list[str] = field(default_factory=list)
    created: list[tuple[str, bool | None]] = field(default_factory=list)

    def whoami(self) -> dict[str, Any]:
        token: dict[str, Any] = {"displayName": "t", "role": self.role}
        if self.fine_grained is not None:
            token["fineGrained"] = self.fine_grained
        return {"type": "user", "name": self.user, "auth": {"type": "access_token", "accessToken": token}}

    def create_repo(self, repo_id: str, *, private: bool | None = None, exist_ok: bool = False) -> str:
        if (self.root / repo_id).exists() and not exist_ok:
            raise FileExistsError(repo_id)
        (self.root / repo_id).mkdir(parents=True, exist_ok=True)
        self.created.append((repo_id, private))
        return f"https://huggingface.co/{repo_id}"

    def _head(self) -> str:
        return f"{len(self.commits):040x}"

    def model_info(self, repo_id: str) -> _Info:
        if self.info_errors:
            raise self.info_errors.pop(0)
        if not (self.root / repo_id).is_dir():
            response = requests.Response()
            response.status_code = 404
            raise RepositoryNotFoundError(f"{repo_id} not found", response=response)
        return _Info(sha=self._head())

    def list_repo_tree(
        self, repo_id: str, *, recursive: bool = False, revision: str | None = None
    ) -> Iterable[RepoFile | RepoFolder]:
        assert recursive
        assert revision == self._head(), "the tree must be read at the commit the caller verifies"
        base = self.root / repo_id
        out: list[RepoFile | RepoFolder] = []
        for path in sorted(p for p in base.rglob("*") if p.is_file()):
            rel = path.relative_to(base).as_posix()
            meta = self.declared[(repo_id, rel)]
            if rel in self.corrupt:
                meta = _declared(path.read_bytes() + b"x")
            lfs = None
            if rel.endswith(self.lfs_suffixes):
                lfs = {"size": meta.size, "oid": meta.sha256, "pointerSize": 134}
            out.append(RepoFile(path=rel, size=meta.size, oid=meta.blob, lfs=lfs))
        return out

    def hf_hub_download(
        self, repo_id: str, filename: str, *, revision: str | None = None, local_dir: str | Path | None = None,
        force_download: bool = False,
    ) -> str:
        assert revision == self._head()
        assert force_download, "a download-back check must never reuse a local copy"
        assert local_dir is not None
        self.downloads.append(filename)
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.root / repo_id / filename, target)
        return str(target)

    def create_commit(
        self,
        repo_id: str,
        operations: Iterable[CommitOperationAdd | CommitOperationDelete],
        *,
        commit_message: str,
        parent_commit: str | None = None,
    ) -> object:
        self.commit_attempts += 1
        if self.commit_error is not None:
            raise self.commit_error
        if self.network_failures > 0:
            self.network_failures -= 1
            raise requests.ConnectionError("simulated network failure")
        if parent_commit != self._head():
            raise RuntimeError(f"stale parent_commit {parent_commit}")
        paths: list[str] = []
        for op in operations:
            if isinstance(op, CommitOperationDelete):
                (self.root / repo_id / op.path_in_repo).unlink()
                del self.declared[(repo_id, op.path_in_repo)]
                paths.append(f"-{op.path_in_repo}")
                continue
            paths.append(op.path_in_repo)
            if op.path_in_repo in self.drop:
                continue
            target = self.root / repo_id / op.path_in_repo
            target.parent.mkdir(parents=True, exist_ok=True)
            src = op.path_or_fileobj
            if isinstance(src, bytes):
                data = src
            elif isinstance(src, str | Path):
                data = Path(src).read_bytes()
            else:
                raise TypeError(f"unexpected path_or_fileobj {type(src).__name__}")
            self.declared[(repo_id, op.path_in_repo)] = _declared(data)
            if op.path_in_repo in self.corrupt_storage:
                data = data + b"\0"
            target.write_bytes(data)
        self.commits.append(paths)
        self.commit_messages.append(commit_message)
        return object()


# ---------------------------------------------------------------------------- fixtures
def _fake_base_fetcher(name: str, revision: str, layout: BestLayout, dest: Path) -> BaseFiles:
    """Stand-in for the Hub download of the base model's card, LICENSE, config and tokenizer files."""
    dest.mkdir(parents=True, exist_ok=True)
    files: dict[str, Path] = {}
    (dest / "LICENSE").write_text(_LICENSE_TEXT)
    files["LICENSE"] = dest / "LICENSE"
    tok_dir = dest / "tok"
    build_tiny_tokenizer().save_pretrained(str(tok_dir))
    files.update({p.name: p for p in tok_dir.iterdir()})
    if layout == "full":
        cfg_dir = dest / "cfg"
        build_tiny_qwen2().config.save_pretrained(str(cfg_dir))
        files["config.json"] = cfg_dir / "config.json"
    return BaseFiles(card=_QWEN_RESEARCH_CARD, files=files)


def _record(qid: str, normal: float, count_once: float, source: str) -> QueryRecord:
    return QueryRecord(
        qid=qid,
        source=source,
        query_text="q",
        rewrite_text="r",
        format_error=None,
        terms=("a",),
        ndcg_normal=normal,
        ndcg_count_once=count_once,
    )


@dataclass(frozen=True)
class _Run:
    run_dir: Path
    runs_dir: Path


def _make_run(
    tmp_path: Path,
    *,
    layout: BestLayout,
    weights: dict[str, torch.Tensor],
    mode: ScoringMode = "normal",
    step: int = 40,
    report_override: str | None = None,
    eval_best_step: int | None = None,
) -> _Run:
    """runs/<mode>/{best/, run_manifest.json, step_000050/ (a resume checkpoint)} and runs/eval/{jsonl, report}."""
    runs = tmp_path / "runs"
    run_dir = runs / mode
    meta = BestMeta(
        step=step,
        val_ndcg_normal=0.4321,
        val_ndcg_count_once=0.4012,
        reward_mode=mode,
        layout=layout,
        config_sha256="c" * 64,
        seeds={"run_seed": 7, "split_seed": 11},
        git_sha="a" * 40,
        git_dirty=False,
    )
    save_best(run_dir, weights=weights, meta=meta)
    resume = run_dir / "step_000050"
    resume.mkdir()
    (resume / "optim.pt").write_bytes(b"optimizer state")
    (resume / "model.safetensors").write_bytes(b"resume weights")
    lora = None
    if layout == "lora":
        lora = {"rank": _LORA.rank, "alpha": _LORA.alpha, "dropout": _LORA.dropout, "target_modules": list(_LORA.target_modules)}
    write_manifest(
        run_dir,
        {
            "format_version": 1,
            "reward_mode": mode,
            "model": {"name": "tiny-local", "revision": "main"},
            "config": {
                "model": {"name": "tiny-local", "revision": "main", "train_mode": layout},
                "lora": lora,
                "run": {"total_steps": 60},
            },
            "config_sha256": "c" * 64,
            "seeds": {"run_seed": 7, "split_seed": 11},
            "git": {"sha": "a" * 40, "dirty": False},
        },
    )
    eval_dir = runs / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    source = f"trained-{mode}"
    records = [_record("1", 0.5, 0.25, source), _record("2", 0.25, 0.5, source)]  # means 0.3750 / 0.3750
    records.append(_record("3", 0.0, 0.0, source))  # means 0.2500 / 0.2500
    header = {"source": source, "trained_steps": 50, "best": {"step": step if eval_best_step is None else eval_best_step}}
    write_records(eval_dir / f"{source}.jsonl", records, header=header)
    report = report_override or (
        "Queries: 3\n\n| source | nDCG@10 normal | nDCG@10 count_once | best step | val score | training steps |\n"
        "|---|---|---|---|---|---|\n"
        f"| trained-normal | 0.2500 | 0.2500 | {step} | 0.4321 (val/ndcg_normal) | 50 |\n"
        f"| trained-count_once | 0.2500 | 0.2500 | {step} | 0.4012 (val/ndcg_count_once) | 50 |\n"
    )
    (eval_dir / "report.md").write_text(report)
    return _Run(run_dir=run_dir, runs_dir=runs)


_LORA_KEYS = {
    "model.layers.0.self_attn.q_proj.lora_A.actor.weight": torch.ones(4, 8),
    "model.layers.0.self_attn.q_proj.lora_B.actor.weight": torch.ones(8, 4),
}


def _publish(hub: FakeHub, run: _Run, **kwargs: Any) -> Any:
    loaded = load_run(run.run_dir)
    return publish_run(
        hub,
        loaded,
        base_fetcher=_fake_base_fetcher,
        namespace=kwargs.pop("namespace", None),
        repo_prefix="drbm25",
        private=True,
        code_url="https://example.org/drbm25",
        sleep=lambda _s: None,
        **kwargs,
    )


# ---------------------------------------------------------------------------- names
def test_model_slug_names_size_and_lora() -> None:
    assert model_slug("Qwen/Qwen2.5-3B-Instruct", "full") == "qwen2.5-3b"
    assert model_slug("Qwen/Qwen2.5-1.5B-Instruct", "lora") == "qwen2.5-1.5b-lora"
    assert model_slug("tiny-local", "full") == "tiny-local"


# ---------------------------------------------------------------------------- file list
def test_upload_list_has_weights_provenance_and_card_but_no_resume_checkpoint(tmp_path: Path) -> None:
    run = _make_run(tmp_path, layout="full", weights={"w": torch.zeros(2)})
    plan = stage_files(
        load_run(run.run_dir), _fake_base_fetcher, tmp_path / "stage", repo_id="u/r", code_url="https://x"
    )
    assert plan["model.safetensors"] == run.run_dir / "best" / "model.safetensors"
    assert plan["provenance/best_meta.json"] == run.run_dir / "best" / "meta.json"
    assert plan["provenance/run_manifest.json"] == run.run_dir / "run_manifest.json"
    assert plan["provenance/trained-normal.jsonl"] == run.runs_dir / "eval" / "trained-normal.jsonl"
    assert plan["provenance/report.md"] == run.runs_dir / "eval" / "report.md"
    assert {"README.md", "LICENSE", "NOTICE", "config.json", "tokenizer.json"} <= plan.keys()
    resume_files = {p.resolve() for p in (run.run_dir / "step_000050").iterdir()}
    assert not resume_files & {p.resolve() for p in plan.values()}
    assert not [k for k in plan if "optim" in k or "step_" in k]


def test_lora_upload_list_is_a_peft_adapter_without_base_config(tmp_path: Path) -> None:
    run = _make_run(tmp_path, layout="lora", weights=dict(_LORA_KEYS))
    plan = stage_files(
        load_run(run.run_dir), _fake_base_fetcher, tmp_path / "stage", repo_id="u/r", code_url="https://x"
    )
    assert {"adapter_model.safetensors", "adapter_config.json", "README.md", "LICENSE"} <= plan.keys()
    assert "model.safetensors" not in plan
    assert "config.json" not in plan
    adapter_config = json.loads(plan["adapter_config.json"].read_text())
    assert (adapter_config["r"], adapter_config["lora_alpha"]) == (4, 8)
    assert adapter_config["base_model_name_or_path"] == "tiny-local"
    assert adapter_config["revision"] == "main"


def test_lora_weights_with_unexpected_names_are_refused(tmp_path: Path) -> None:
    run = _make_run(tmp_path, layout="lora", weights={"model.layers.0.q_proj.weight": torch.ones(2)})
    with pytest.raises(ValueError, match="not an actor LoRA weight"):
        stage_files(load_run(run.run_dir), _fake_base_fetcher, tmp_path / "s", repo_id="u/r", code_url="https://x")


# ---------------------------------------------------------------------------- card
def test_model_card_contains_the_fixture_numbers(tmp_path: Path) -> None:
    run = _make_run(tmp_path, layout="full", weights={"w": torch.zeros(2)}, mode="count_once", step=40)
    card = build_model_card(
        load_run(run.run_dir), _QWEN_RESEARCH_CARD, repo_id="tester/drbm25-x", code_url="https://example.org/drbm25"
    )
    assert "| validation (best step 40) | 0.4321 | 0.4012 |" in card
    assert "| test (3 queries) | 0.2500 | 0.2500 |" in card
    assert "reward mode: `count_once`" in card
    assert "trained steps: 50 (planned 60)" in card
    assert "tiny-local@main" in card
    assert "c" * 64 in card
    assert '"run_seed": 7' in card
    assert f"https://example.org/drbm25/tree/{'a' * 40}" in card
    assert "license: other\nlicense_name: qwen-research\n" in card
    assert "Built with Qwen" in card


def test_report_that_disagrees_with_the_eval_file_is_refused(tmp_path: Path) -> None:
    bad = "| trained-normal | 0.9999 | 0.2500 | 40 | x | 50 |\n"
    run = _make_run(tmp_path, layout="full", weights={"w": torch.zeros(2)}, report_override=bad)
    with pytest.raises(ValueError, match="report"):
        load_run(run.run_dir)


def test_eval_of_another_best_step_is_refused(tmp_path: Path) -> None:
    run = _make_run(tmp_path, layout="full", weights={"w": torch.zeros(2)}, eval_best_step=30)
    with pytest.raises(ValueError, match="best step"):
        load_run(run.run_dir)


# ---------------------------------------------------------------------------- token
def test_read_only_token_fails_before_any_upload(tmp_path: Path) -> None:
    hub = FakeHub(tmp_path / "hub", role="read")
    run = _make_run(tmp_path, layout="full", weights={"w": torch.zeros(2)})
    with pytest.raises(TokenPermissionError, match=r"read-only"):
        _publish(hub, run)
    assert hub.created == []
    assert hub.commits == []
    assert not (run.run_dir / UPLOADED_FILE).exists()


def test_fine_grained_token_needs_repo_write_on_the_user() -> None:
    def who(perms: list[str]) -> dict[str, Any]:
        token = {"role": "fineGrained", "fineGrained": {"scoped": [{"entity": {"type": "user", "name": "u"}, "permissions": perms}]}}
        return {"name": "u", "auth": {"accessToken": token}}

    assert check_write_access(who(["repo.content.read", "repo.write"])) == "u"
    with pytest.raises(TokenPermissionError, match=r"repo\.write"):
        check_write_access(who(["repo.content.read"]))
    with pytest.raises(TokenPermissionError):
        check_write_access({"name": "u", "auth": {"type": "oauth"}})


# ---------------------------------------------------------------------------- upload + verify
def test_publish_uploads_verifies_and_writes_the_marker(tmp_path: Path) -> None:
    hub = FakeHub(tmp_path / "hub")
    run = _make_run(tmp_path, layout="full", weights={"w": torch.zeros(2)})
    result = _publish(hub, run)
    assert result.repo_id == "tester/drbm25-tiny-local-normal"
    assert hub.created == [(result.repo_id, True)]
    assert len(hub.commits) == 2  # the files, then UPLOADED.json
    assert hub.downloads == ["model.safetensors"]  # every LFS (weight) file is downloaded back and hashed
    assert "best step 40" in hub.commit_messages[0]
    assert "a" * 40 in hub.commit_messages[0]
    marker = json.loads((run.run_dir / UPLOADED_FILE).read_text())
    assert marker["repo_id"] == result.repo_id
    assert marker["revision"] == result.revision
    local = (run.run_dir / "best" / "model.safetensors").read_bytes()
    assert marker["files"]["model.safetensors"] == {"sha256": hashlib.sha256(local).hexdigest(), "size": len(local)}
    assert "verified_at" in marker
    assert (hub.root / result.repo_id / UPLOADED_FILE).is_file()


def test_sha_mismatch_fails_verification_and_writes_no_marker(tmp_path: Path) -> None:
    hub = FakeHub(tmp_path / "hub", corrupt={"model.safetensors"})
    run = _make_run(tmp_path, layout="full", weights={"w": torch.zeros(2)})
    with pytest.raises(VerificationError, match=r"model\.safetensors"):
        _publish(hub, run)
    assert not (run.run_dir / UPLOADED_FILE).exists()


def test_regular_file_mismatch_is_caught_by_its_git_blob_id(tmp_path: Path) -> None:
    hub = FakeHub(tmp_path / "hub", corrupt={"README.md"})
    run = _make_run(tmp_path, layout="full", weights={"w": torch.zeros(2)})
    with pytest.raises(VerificationError, match=r"README\.md"):
        _publish(hub, run)
    assert not (run.run_dir / UPLOADED_FILE).exists()


def test_missing_remote_file_fails_verification_and_writes_no_marker(tmp_path: Path) -> None:
    hub = FakeHub(tmp_path / "hub", drop={"provenance/report.md"})
    run = _make_run(tmp_path, layout="full", weights={"w": torch.zeros(2)})
    with pytest.raises(VerificationError, match=r"provenance/report\.md"):
        _publish(hub, run)
    assert not (run.run_dir / UPLOADED_FILE).exists()


def test_rerun_skips_files_already_on_the_hub(tmp_path: Path) -> None:
    hub = FakeHub(tmp_path / "hub")
    run = _make_run(tmp_path, layout="full", weights={"w": torch.zeros(2)})
    first = _publish(hub, run)
    commits_after_first = len(hub.commits)
    second = _publish(hub, run)
    # No file is uploaded again; only the marker is withdrawn at the start and re-committed after verification.
    assert hub.commits[commits_after_first:] == [[f"-{UPLOADED_FILE}"], [UPLOADED_FILE]]
    assert second.uploaded == []
    assert second.repo_id == first.repo_id
    assert (run.run_dir / UPLOADED_FILE).is_file()


def test_rerun_uploads_only_the_changed_file(tmp_path: Path) -> None:
    hub = FakeHub(tmp_path / "hub")
    run = _make_run(tmp_path, layout="full", weights={"w": torch.zeros(2)})
    _publish(hub, run)
    (run.runs_dir / "eval" / "report.md").write_text(
        (run.runs_dir / "eval" / "report.md").read_text() + "\nextra line\n"
    )
    result = _publish(hub, run)
    assert result.uploaded == ["provenance/report.md"]


def test_publish_withdraws_both_markers_before_uploading(tmp_path: Path) -> None:
    hub = FakeHub(tmp_path / "hub")
    run = _make_run(tmp_path, layout="full", weights={"w": torch.zeros(2)})
    first = _publish(hub, run)
    report = run.runs_dir / "eval" / "report.md"
    report.write_text(report.read_text() + "\nextra line\n")
    hub.drop = {"provenance/report.md"}  # this upload never verifies
    with pytest.raises(VerificationError):
        _publish(hub, run)
    assert hub.commits[2] == [f"-{UPLOADED_FILE}"]  # withdrawn before anything was staged or uploaded
    assert not (hub.root / first.repo_id / UPLOADED_FILE).exists()
    assert not (run.run_dir / UPLOADED_FILE).exists()


def test_corrupted_stored_weights_with_a_correct_declared_hash_fail_the_download_check(tmp_path: Path) -> None:
    hub = FakeHub(tmp_path / "hub", corrupt_storage={"model.safetensors"})
    run = _make_run(tmp_path, layout="full", weights={"w": torch.zeros(2)})
    with pytest.raises(VerificationError, match=r"model\.safetensors.*downloaded"):
        _publish(hub, run)
    assert not (run.run_dir / UPLOADED_FILE).exists()
    assert not (hub.root / "tester/drbm25-tiny-local-normal" / UPLOADED_FILE).exists()
    assert not list(run.run_dir.glob(".tmp-download*"))  # the downloaded copy is removed


def test_invalidate_withdraws_the_hub_and_local_markers(tmp_path: Path) -> None:
    hub = FakeHub(tmp_path / "hub")
    run = _make_run(tmp_path, layout="full", weights={"w": torch.zeros(2)})
    result = _publish(hub, run)
    invalidate_run(hub, run.run_dir, namespace=None, repo_prefix="drbm25", sleep=lambda _s: None)
    assert not (run.run_dir / UPLOADED_FILE).exists()
    assert not (hub.root / result.repo_id / UPLOADED_FILE).exists()
    assert (hub.root / result.repo_id / "model.safetensors").is_file()  # only the marker goes


def test_invalidate_without_a_repo_or_a_manifest_changes_nothing(tmp_path: Path) -> None:
    hub = FakeHub(tmp_path / "hub")
    run = _make_run(tmp_path, layout="full", weights={"w": torch.zeros(2)})
    invalidate_run(hub, run.run_dir, namespace=None, repo_prefix="drbm25", sleep=lambda _s: None)  # no repo yet
    empty = tmp_path / "runs-empty" / "normal"
    empty.mkdir(parents=True)
    invalidate_run(hub, empty, namespace=None, repo_prefix="drbm25", sleep=lambda _s: None)  # nothing trained yet
    assert hub.commits == []
    assert hub.created == []


def test_network_errors_are_retried_a_bounded_number_of_times(tmp_path: Path) -> None:
    hub = FakeHub(tmp_path / "hub", network_failures=2)
    run = _make_run(tmp_path, layout="full", weights={"w": torch.zeros(2)})
    _publish(hub, run)
    assert (run.run_dir / UPLOADED_FILE).is_file()

    hub_down = FakeHub(tmp_path / "hub2", network_failures=100)
    run2 = _make_run(tmp_path / "second", layout="full", weights={"w": torch.zeros(2)})
    with pytest.raises(requests.ConnectionError):
        _publish(hub_down, run2)
    assert hub_down.network_failures == 100 - 4  # 4 attempts, then the error propagates
    assert not (run2.run_dir / UPLOADED_FILE).exists()


def test_dry_run_uploads_nothing(tmp_path: Path) -> None:
    hub = FakeHub(tmp_path / "hub")
    run = _make_run(tmp_path, layout="full", weights={"w": torch.zeros(2)})
    assert _publish(hub, run, dry_run=True) is None
    assert hub.created == []
    assert hub.commits == []
    assert not (run.run_dir / UPLOADED_FILE).exists()


def test_volume_is_safe_only_when_both_runs_are_uploaded_and_current(tmp_path: Path) -> None:
    hub = FakeHub(tmp_path / "hub")
    normal = _make_run(tmp_path, layout="full", weights={"w": torch.zeros(2)}, mode="normal")
    _publish(hub, normal)
    assert not volume_safe(normal.runs_dir)
    count_once = _make_run(tmp_path, layout="full", weights={"w": torch.zeros(2)}, mode="count_once")
    _publish(hub, count_once)
    assert volume_safe(normal.runs_dir)
    # A newer best/ after the upload makes the marker stale.
    meta_path = count_once.run_dir / "best" / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta_path.write_text(json.dumps({**meta, "step": 99}))
    assert not volume_safe(normal.runs_dir)


# ---------------------------------------------------------------------------- round trips
def _logits(model: torch.nn.Module, ids: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits


def test_full_layout_round_trip_loads_with_from_pretrained(tmp_path: Path) -> None:
    model = build_tiny_qwen2(seed=3).eval()
    run = _make_run(tmp_path, layout="full", weights=dict(model.state_dict()))
    hub = FakeHub(tmp_path / "hub")
    result = _publish(hub, run)
    loaded = AutoModelForCausalLM.from_pretrained(str(hub.root / result.repo_id)).eval()
    ids = torch.tensor([[3, 4, 5, 6, 7]])
    assert torch.equal(_logits(loaded, ids), _logits(model, ids))


def test_lora_round_trip_loads_with_peft(tmp_path: Path) -> None:
    models = LoraPolicyModels(build_tiny_qwen2(seed=5), _LORA, gradient_checkpointing=False)
    with torch.no_grad():  # LoRA B starts at zero; make the adapter change the output
        for p in models.actor_parameters():
            p.add_(0.05 * torch.randn_like(p))
    run = _make_run(tmp_path, layout="lora", weights=models.actor_state_dict())
    hub = FakeHub(tmp_path / "hub")
    result = _publish(hub, run)
    peft_model = PeftModel.from_pretrained(build_tiny_qwen2(seed=5), str(hub.root / result.repo_id)).eval()
    ids = torch.tensor([[3, 4, 5, 6, 7]])
    expected = _logits(models.generation_module(), ids)
    assert not torch.equal(expected, _logits(build_tiny_qwen2(seed=5).eval(), ids))
    assert torch.equal(_logits(peft_model, ids), expected)


# ---------------------------------------------------------------------------- errors and CLI
def _http_error(status: int) -> HfHubHTTPError:
    response = requests.Response()
    response.status_code = status
    return HfHubHTTPError(f"HTTP {status}", response=response)


def test_forbidden_commit_is_not_retried(tmp_path: Path) -> None:
    hub = FakeHub(tmp_path / "hub", commit_error=_http_error(403))
    run = _make_run(tmp_path, layout="full", weights={"w": torch.zeros(2)})
    with pytest.raises(HfHubHTTPError):
        _publish(hub, run)
    assert hub.commit_attempts == 1
    assert not (run.run_dir / UPLOADED_FILE).exists()


def test_server_errors_are_retried(tmp_path: Path) -> None:
    hub = FakeHub(tmp_path / "hub", info_errors=[_http_error(503), _http_error(502)])
    run = _make_run(tmp_path, layout="full", weights={"w": torch.zeros(2)})
    _publish(hub, run)
    assert hub.info_errors == []
    assert (run.run_dir / UPLOADED_FILE).is_file()


def _cli(argv: list[str]) -> int:
    try:
        return publish_main(argv)
    finally:
        shutdown()


def test_cli_says_safe_to_delete_only_after_both_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    hub = FakeHub(tmp_path / "hub")
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: hub)
    monkeypatch.setattr(publish_module, "hub_base_fetcher", _fake_base_fetcher)
    normal = _make_run(tmp_path, layout="full", weights={"w": torch.zeros(2)}, mode="normal")
    count_once = _make_run(tmp_path, layout="full", weights={"w": torch.zeros(2)}, mode="count_once")

    assert _cli(["--run-dir", str(normal.run_dir)]) == 0
    first = capsys.readouterr().out
    assert "tester/drbm25-tiny-local-normal" in first
    assert "safe to delete the volume" not in first

    assert _cli(["--run-dir", str(count_once.run_dir), "--public"]) == 0
    assert "safe to delete the volume" in capsys.readouterr().out
    assert hub.created[-1] == ("tester/drbm25-tiny-local-count_once", False)

    assert _cli(["--run-dir", str(normal.run_dir), "--invalidate"]) == 0
    assert not (normal.run_dir / UPLOADED_FILE).exists()
    assert not (hub.root / "tester/drbm25-tiny-local-normal" / UPLOADED_FILE).exists()
