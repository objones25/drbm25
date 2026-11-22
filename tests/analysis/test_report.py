from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import wandb

import drbm25.analysis.report as report_mod
from drbm25.analysis.report import build_report, load_outputs, log_analysis, make_figures, summary_markdown
from drbm25.config import load_config

REPO = Path(__file__).resolve().parents[2]
SMOKE = REPO / "configs" / "smoke_cpu.yaml"


@dataclass
class _FakeRun:
    entity: str = "ent"
    project: str = "proj"
    id: str = "abc123"
    logged: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[Any] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def log(self, data: dict[str, Any]) -> None:
        self.logged.append(data)

    def log_artifact(self, artifact: Any) -> None:
        self.artifacts.append(artifact)


def test_summary_markdown_reports_primary_result_and_means(metrics_dir: Path) -> None:
    text = summary_markdown(load_outputs(metrics_dir))

    assert "trained-normal" in text
    assert "trained-count_once" in text
    assert "nDCG@10" in text
    assert "95% CI" in text


def test_make_figures_writes_pngs(metrics_dir: Path, tmp_path: Path) -> None:
    figures = make_figures(load_outputs(metrics_dir), tmp_path / "fig")

    assert "figures/ndcg_ecdf" in figures
    assert all(p.suffix == ".png" and p.stat().st_size > 0 for p in figures.values())


def test_log_analysis_logs_tables_figures_scalars_and_artifact(
    metrics_dir: Path, db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _FakeRun()
    opened: dict[str, Any] = {}

    @contextmanager
    def fake_open_run(cfg: object, **kwargs: Any) -> Generator[_FakeRun]:
        opened.update(kwargs)
        yield run

    monkeypatch.setattr(report_mod, "open_run", fake_open_run)

    path = log_analysis(load_config(SMOKE), metrics_dir, db, work=tmp_path / "w", browse_rows=6)

    assert path == "ent/proj/abc123"
    assert opened["job_type"] == "analysis"
    assert opened["run_name"] == "rewrite-analysis"
    logged = {k: v for row in run.logged for k, v in row.items()}
    wanted = ("tables/query_metrics", "tables/comparisons", "tables/browse", "figures/ndcg_ecdf")
    assert [k for k in wanted if k not in logged] == []
    assert isinstance(logged["tables/browse"], wandb.Table)
    assert len(logged["tables/browse"].data) == 6
    assert any(k.startswith("summary/trained-normal/") for k in run.summary)
    (artifact,) = run.artifacts
    assert artifact.type == "dataset"


class _FakeWr:
    """Records what build_report constructs; mirrors the wandb_workspaces.reports.v2 names it uses."""

    def __init__(self) -> None:
        self.saved: list[Any] = []

        def node(kind: str) -> Any:
            return lambda *args, **kwargs: SimpleNamespace(kind=kind, args=args, **kwargs)

        for kind in ("H1", "H2", "P", "MarkdownBlock", "PanelGrid", "Runset", "MediaBrowser", "WeavePanelSummaryTable"):
            setattr(self, kind, node(kind))
        outer = self

        class Report(SimpleNamespace):
            url = "https://wandb.ai/ent/proj/reports/x"

            def save(self) -> None:
                outer.saved.append(self)

        self.Report = Report


def test_build_report_has_sections_filtered_to_the_analysis_run(metrics_dir: Path) -> None:
    wr = _FakeWr()

    url = build_report("ent", "proj", "rewrite-analysis", load_outputs(metrics_dir), wr=wr)

    assert url == "https://wandb.ai/ent/proj/reports/x"
    (report,) = wr.saved
    headings = [b.text for b in report.blocks if b.kind == "H2"]
    assert headings == [
        "Summary",
        "Retrieval quality",
        "Rewrite behavior",
        "Diversity",
        "Vocabulary",
        "Reasoning",
        "Methods",
    ]
    grids = [b for b in report.blocks if b.kind == "PanelGrid"]
    assert grids
    assert all(g.runsets[0].query == "rewrite-analysis" for g in grids)
