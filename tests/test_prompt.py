from pathlib import Path

from drbm25.prompt import build_prompt

GOLDEN = Path(__file__).parent / "golden" / "prompt_polymers.txt"


def test_prompt_matches_upstream_bytes():
    expected = GOLDEN.read_bytes().decode("utf-8")
    assert build_prompt("how are polymers made") == expected


def test_prompt_ends_inside_think():
    assert build_prompt("any query").endswith("Let me rewrite the query with reasoning. \n<think>\n")
