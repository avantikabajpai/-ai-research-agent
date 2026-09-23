"""
eval.py — Automated evaluation harness for the agent.

WHY THIS MATTERS FOR AN INTERVIEW:
Anyone can demo an agent answering one cherry-picked question well. This file
runs a fixed set of test questions with known-correct answers every time you
change anything (a prompt, a tool, the model), and reports pass/fail plus
timing — so you can say "I measured this" instead of "it seemed to work."
This is standard ML engineering practice: you don't ship changes to a model
or prompt without re-running your eval set.

HOW SCORING WORKS:
Each test case defines `expect_contains`: substrings that should appear
somewhere in the agent's final answer (case-insensitive). This is a simple,
transparent scoring method — no external judge model needed — appropriate
for questions with objectively checkable answers (arithmetic, known facts).
It intentionally does NOT check exact wording, since the same correct answer
can be phrased many ways.
"""

import time
from dotenv import load_dotenv

load_dotenv()

from agent import run_agent  # noqa: E402  (import after load_dotenv on purpose)

TEST_CASES = [
    {
        "name": "basic_arithmetic",
        "question": "What is 47 * 6?",
        "expect_contains": ["282"],
    },
    {
        "name": "arithmetic_with_parens",
        "question": "Calculate (15 + 5) * 3",
        "expect_contains": ["60"],
    },
    {
        "name": "requires_web_search",
        "question": "What year was the Eiffel Tower completed?",
        "expect_contains": ["1889"],
    },
    {
        "name": "multi_tool_chain",
        "question": "If a country's population is 45 million and 12% are under 10, how many people is that?",
        "expect_contains": ["5.4", "5,400,000"],
    },
    {
        "name": "declines_to_guess",
        "question": "What will the Bitcoin price be exactly one year from today?",
        "expect_contains": ["can't", "cannot", "unable", "uncertain", "don't know", "no way to know", "impossible to"],
    },
]


def run_eval(verbose_agent: bool = False) -> None:
    results = []

    for case in TEST_CASES:
        start = time.time()
        try:
            answer = run_agent(case["question"], verbose=verbose_agent)
            elapsed = time.time() - start
            answer_lower = answer.lower()
            passed = any(kw.lower() in answer_lower for kw in case["expect_contains"])
        except Exception as e:
            elapsed = time.time() - start
            answer = f"ERROR: {e}"
            passed = False

        results.append({"name": case["name"], "passed": passed, "elapsed": elapsed, "answer": answer})

        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {case['name']} ({elapsed:.1f}s)")
        if not passed:
            print(f"       Question: {case['question']}")
            print(f"       Got: {answer[:200]}")

    passed_count = sum(r["passed"] for r in results)
    total = len(results)
    avg_time = sum(r["elapsed"] for r in results) / total if total else 0

    print("\n" + "=" * 40)
    print(f"Score: {passed_count}/{total} passed ({passed_count/total*100:.0f}%)")
    print(f"Avg response time: {avg_time:.1f}s")
    print("=" * 40)


if __name__ == "__main__":
    run_eval()
