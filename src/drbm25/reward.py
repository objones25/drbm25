"""Format check, nDCG@k and the scalar reward for the query-rewriting task.

Mirrors upstream DeepRetrieval (``msmarco.py`` format checks and reward,
``Lucene/utils.py`` nDCG) with two deliberate differences: a non-string
``query`` is a format error, and no exception is swallowed.
"""

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

REWARD_FORMAT_ERROR: Final = -2.0
FORMAT_BONUS: Final = 0.1
_REWARD_MAX: Final = FORMAT_BONUS + 1.0

_THINK_OPEN: Final = "<think>"
_THINK_CLOSE: Final = "</think>"
_ANSWER_OPEN: Final = "<answer>"
_ANSWER_CLOSE: Final = "</answer>"


class FormatErrorReason(StrEnum):
    THINK_REOPENED = "think_reopened"
    TAG_COUNT = "tag_count"
    TAG_ORDER = "tag_order"
    BAD_JSON = "bad_json"
    NO_QUERY_KEY = "no_query_key"
    QUERY_NOT_STRING = "query_not_string"


@dataclass(frozen=True)
class ParsedRewrite:
    query: str


@dataclass(frozen=True)
class FormatError:
    reason: FormatErrorReason


def parse_response(response_text: str) -> ParsedRewrite | FormatError:
    """Validate the generated text and extract the rewritten query.

    ``response_text`` is the generated text only; the prompt already ends with
    ``<think>\\n``. Upstream requires exactly one ``<think>`` in prompt plus
    response, so the response must contain none, and exactly one each of
    ``</think>``, ``<answer>``, ``</answer>``, in that order.

    When several problems exist, the first that applies is reported, in this
    order: THINK_REOPENED, TAG_COUNT, TAG_ORDER, then the JSON reasons
    (BAD_JSON, NO_QUERY_KEY, QUERY_NOT_STRING).
    """
    if _THINK_OPEN in response_text:
        return FormatError(FormatErrorReason.THINK_REOPENED)
    tags = (_THINK_CLOSE, _ANSWER_OPEN, _ANSWER_CLOSE)
    if any(response_text.count(tag) != 1 for tag in tags):
        return FormatError(FormatErrorReason.TAG_COUNT)
    think_close, answer_open, answer_close = (response_text.find(tag) for tag in tags)
    if not think_close < answer_open < answer_close:
        return FormatError(FormatErrorReason.TAG_ORDER)

    body = response_text[answer_open + len(_ANSWER_OPEN) : answer_close].strip()
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return FormatError(FormatErrorReason.BAD_JSON)
    if not isinstance(data, dict) or "query" not in data:
        return FormatError(FormatErrorReason.NO_QUERY_KEY)
    query = data["query"]
    if not isinstance(query, str):
        return FormatError(FormatErrorReason.QUERY_NOT_STRING)
    return ParsedRewrite(query)


def ndcg_at_k(ranked_ids: Sequence[str], relevant_ids: frozenset[str], k: int) -> float:
    """nDCG@k with binary gains; DCG is the sum of 1/log2(rank+1), rank from 1."""
    if not relevant_ids:
        raise ValueError("relevant_ids is empty")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    top = ranked_ids[:k]
    hits = [rank for rank, doc_id in enumerate(top, start=1) if doc_id in relevant_ids]
    if len({top[rank - 1] for rank in hits}) != len(hits):
        raise ValueError("ranked_ids repeats a relevant document; a document can only be ranked once")
    dcg = sum(1.0 / math.log2(rank + 1) for rank in hits)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant_ids), k) + 1))
    return dcg / idcg


def compute_reward(parsed: ParsedRewrite | FormatError, ndcg: float | None) -> float:
    """-2 for a format error; 0 when nDCG is 0; otherwise 0.1 + nDCG."""
    if isinstance(parsed, FormatError):
        return REWARD_FORMAT_ERROR
    if ndcg is None or not 0.0 <= ndcg <= 1.0:
        raise ValueError(f"ndcg must be in [0, 1] for a parsed response, got {ndcg!r}")
    reward = FORMAT_BONUS + ndcg if ndcg > 0.0 else 0.0
    if not REWARD_FORMAT_ERROR <= reward <= _REWARD_MAX:
        raise AssertionError(f"reward {reward} outside [{REWARD_FORMAT_ERROR}, {_REWARD_MAX}]")
    return reward
