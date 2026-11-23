import pytest
from hypothesis import given
from hypothesis import strategies as st

from drbm25.reward import (
    REWARD_FORMAT_ERROR,
    FormatError,
    FormatErrorReason,
    ParsedRewrite,
    compute_reward,
    ndcg_at_k,
    parse_response,
    think_text,
)

R = FormatErrorReason


def test_parses_valid_response():
    text = 'reason\n</think>\n<answer>\n{"query": "polymer synthesis"}\n</answer>'
    assert parse_response(text) == ParsedRewrite("polymer synthesis")


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ('a <think> b</think><answer>{"query": "x"}</answer>', R.THINK_REOPENED),
        ('a</think><answer>{"query": "x"}', R.TAG_COUNT),
        ('a</think><answer>{"query": "x"}</answer><answer>{"query": "y"}</answer>', R.TAG_COUNT),
        ('<answer>{"query": "x"}</answer> a</think>', R.TAG_ORDER),
        ("a</think><answer>{query: x}</answer>", R.BAD_JSON),
        ("a</think><answer>  </answer>", R.BAD_JSON),
        ('a</think><answer>{"q": "x"}</answer>', R.NO_QUERY_KEY),
        ('a</think><answer>["query"]</answer>', R.NO_QUERY_KEY),
        ('a</think><answer>{"query": ["x"]}</answer>', R.QUERY_NOT_STRING),
        # two problems: precedence is THINK_REOPENED > TAG_COUNT > TAG_ORDER > JSON
        ("<think></think><answer>{bad</answer></answer>", R.THINK_REOPENED),
        ("</answer><answer></think></think>{bad", R.TAG_COUNT),
        ("<answer>{bad</answer></think>", R.TAG_ORDER),
    ],
)
def test_rejects_malformed_response(text, reason):
    assert parse_response(text) == FormatError(reason)


def test_answer_after_end_tag_order_is_checked_between_answer_tags():
    assert parse_response('</think></answer>{"query": "x"}<answer>') == FormatError(R.TAG_ORDER)


@given(st.text())
def test_parse_never_raises(text):
    result = parse_response(text)
    assert isinstance(result, FormatError | ParsedRewrite)
    # a successful parse implies the tag structure upstream requires
    assert isinstance(result, FormatError) or (text.count("</think>") == 1 and "<think>" not in text)


@pytest.mark.parametrize(
    ("ranked", "relevant", "expected"),
    [
        (["a"], {"a"}, 1.0),
        (["x", "y", "a"], {"a"}, 0.5),
        (["x"] * 10 + ["a"], {"a"}, 0.0),
        (["a", "x", "b"], {"a", "b"}, pytest.approx(0.919720, abs=1e-6)),
        ([], {"a"}, 0.0),
    ],
)
def test_ndcg_hand_values(ranked, relevant, expected):
    assert ndcg_at_k(ranked, frozenset(relevant), 10) == expected


def test_ndcg_ideal_capped_at_k():
    # 3 relevant, k=2: ideal has 2 positions, so perfect top-2 scores 1.0
    assert ndcg_at_k(["a", "b"], frozenset({"a", "b", "c"}), 2) == pytest.approx(1.0)


def test_ndcg_rejects_duplicate_ranked_ids():
    with pytest.raises(ValueError, match="repeats a relevant document"):
        ndcg_at_k(["a", "x", "a"], frozenset({"a"}), 10)


@given(
    st.lists(st.sampled_from("abcdef"), max_size=6, unique=True),
    st.frozensets(st.sampled_from("abcdef"), min_size=1),
    st.integers(min_value=1, max_value=15),
)
def test_ndcg_bounds(ranked, relevant, k):
    assert 0.0 <= ndcg_at_k(ranked, relevant, k) <= 1.0 + 1e-12


def test_ndcg_rejects_empty_relevant_set():
    with pytest.raises(ValueError, match="relevant_ids is empty"):
        ndcg_at_k(["a"], frozenset(), 10)


def test_ndcg_rejects_nonpositive_k():
    with pytest.raises(ValueError, match="k must be positive"):
        ndcg_at_k(["a"], frozenset({"a"}), 0)


@pytest.mark.parametrize(
    ("parsed", "ndcg", "expected"),
    [
        (FormatError(R.BAD_JSON), None, -2.0),
        (ParsedRewrite("q"), 0.0, 0.0),
        (ParsedRewrite("q"), 0.5, pytest.approx(0.6)),
        (ParsedRewrite("q"), 1.0, pytest.approx(1.1)),
    ],
)
def test_reward_cases(parsed, ndcg, expected):
    assert compute_reward(parsed, ndcg) == expected


def test_reward_constant():
    assert pytest.approx(-2.0) == REWARD_FORMAT_ERROR


def test_reward_requires_ndcg_for_parsed():
    with pytest.raises(ValueError, match="ndcg"):
        compute_reward(ParsedRewrite("q"), None)


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_reward_rejects_out_of_range_ndcg(bad):
    with pytest.raises(ValueError, match="ndcg"):
        compute_reward(ParsedRewrite("q"), bad)


def test_think_text_returns_stripped_reasoning():
    assert think_text('  plan here \n</think>\n<answer>{"query": "x"}</answer>') == "plan here"


def test_think_text_none_without_close_tag():
    assert think_text('no tags <answer>{"query": "x"}</answer>') is None


def test_think_text_none_with_two_close_tags():
    assert think_text("a</think>b</think>") is None
