"""Prompt templates for the ReAct agent.

Three variants each for single- and multi-objective tasks. The variants exist so
prompt sensitivity can be *reported as a finding* (validity section of the plan),
not hidden. ``v0`` is the default and is modelled on the Search-R1 / MEM1 prompts
in this repo. The action space (<think>/<search>/<answer>, results in
<information>) is identical across variants, matching Search-R1.
"""

SINGLE = {
    "v0": (
        "Answer the question. You must reason inside <think> and </think> first. "
        "If you need facts, search with <search> keywords </search>; the top results "
        "will appear inside <information> and </information>. You may search multiple "
        "times. When you have enough information, give the final answer inside "
        "<answer> and </answer> using only the essential words, e.g. <answer> Beijing "
        "</answer>.\nQuestion: {questions}\n"
    ),
    "v1": (  # terse
        "Solve the question with reasoning and search.\n"
        "Format: <think>...</think> then either <search>query</search> or "
        "<answer>short answer</answer>. Search results come back in "
        "<information>...</information>.\nQuestion: {questions}\n"
    ),
    "v2": (  # verbose
        "You are a careful research assistant answering a factual question. Think "
        "step by step inside <think> and </think>. Whenever you are missing a fact, "
        "issue a search query inside <search> and </search> and read the passages "
        "returned inside <information> and </information>. Search as many times as you "
        "need, one query at a time. Once you are confident, respond with the final, "
        "concise answer inside <answer> and </answer> (only the essential words).\n"
        "Question: {questions}\n"
    ),
}

MULTI = {
    "v0": (
        "You will answer multiple questions using iterative reasoning and search. "
        "Reason inside <think> and </think>. To gather facts, issue ONE query at a "
        "time inside <search> and </search>; results appear inside <information> and "
        "</information>. When every question is answered, provide all final answers, "
        "separated by semicolons, inside <answer> answer1; answer2; ... </answer>. "
        "Each answer must be concise -- only the essential words.\n"
        "Answer the following questions: {questions}\n"
    ),
    "v1": (  # terse
        "Answer all the questions below using reasoning and search.\n"
        "Format: <think>...</think> then <search>query</search> (one at a time) or "
        "<answer>a1; a2; ...</answer>. Results come in <information>...</information>.\n"
        "Questions: {questions}\n"
    ),
    "v2": (  # verbose
        "You are answering several factual questions at once. Maintain your reasoning "
        "inside <think> and </think>. Search for missing facts ONE query at a time "
        "inside <search> and </search>; passages return inside <information> and "
        "</information>. Only once ALL questions can be answered, output every answer "
        "in order, separated by semicolons, inside <answer> ... </answer>, each answer "
        "concise.\nAnswer the following questions: {questions}\n"
    ),
}

CONTINUE_CUE = (
    "\nNow produce your next step: a <think>...</think> followed by exactly one "
    "<search>...</search> or <answer>...</answer>."
)

FORCE_ANSWER_CUE = (
    "\nYou must now answer using only the information above. Do NOT reason further. "
    "Output only the final answer(s) inside <answer> and </answer>, nothing else."
)


def instruction(n_objectives: int, variant: str, questions: str) -> str:
    table = SINGLE if n_objectives <= 1 else MULTI
    if variant not in table:
        raise ValueError(f"unknown prompt variant {variant!r}; have {list(table)}")
    return table[variant].format(questions=questions)
