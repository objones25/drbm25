"""The training prompt, byte-identical to upstream DeepRetrieval's sparse-mode prompt.

Source: upstream/DeepRetrieval/code/data_preprocess/msmarco_sub.py, `make_prefix`
(retrieval_mode="sparse"). Explicit `\\n` escapes keep the trailing spaces upstream has.
"""

_INSTRUCTION_SPARSE = (
    "\nYou are a query rewriting expert. Your task is to augment the user query to find "
    "relevant literature in a corpus with sparse retrieval.\n"
)

_SYSTEM_AND_USER_HEADER = (
    "<|im_start|>system\nYou are a helpful assistant. You first thinks about the reasoning "
    "process in the mind and then provides the user with the answer.<|im_end|>\n<|im_start|>user\n"
)

_FORMAT_INSTRUCTIONS = (
    "\nShow your work in <think> </think> tags. Your final response must be in JSON format "
    "within <answer> </answer> tags. For example,\n"
    '<think>\n[thinking process]\n</think>\n<answer>\n{\n    "query": "...."\n} \n</answer>. \n'
)

_SPARSE_NOTE = (
    'Note: The query should use Boolean operators (AND, OR) and parentheses for grouping terms appropriately."\n'
)

_QUERY_INTRO = "\nHere's the user query:\n"

_ASSISTANT_TURN = "<|im_end|>\n<|im_start|>assistant\n\nLet me rewrite the query with reasoning. \n<think>\n"


def build_prompt(query: str) -> str:
    return (
        _SYSTEM_AND_USER_HEADER
        + _INSTRUCTION_SPARSE
        + _FORMAT_INSTRUCTIONS
        + _SPARSE_NOTE
        + _QUERY_INTRO
        + query
        + _ASSISTANT_TURN
    )
