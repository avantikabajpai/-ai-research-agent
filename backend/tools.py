"""
tools.py — Defines what the agent can DO, beyond just talking.

Two things live here for every tool:
1. A JSON-schema description (the `TOOLS` list) — this is what gets sent to
   Groq so the model knows the tool exists, what it's for, and what
   arguments it takes.
2. The actual Python function that runs when the model decides to call it
   (the `TOOL_FUNCTIONS` dict) — this is OUR code, not the model's.

The model never executes anything itself. It only ever outputs "please call
web_search with query=X" as structured JSON — we (agent.py) are the ones who
actually run it and hand the result back. This separation is the whole safety
model behind tool-using agents: the model proposes, your code decides whether
to execute.
"""

import os
from tavily import TavilyClient

tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


def web_search(query: str) -> str:
    """Searches the web via Tavily and returns a condensed set of results."""
    response = tavily_client.search(query=query, max_results=4)
    results = response.get("results", [])
    if not results:
        return "No results found."

    formatted = []
    for r in results:
        formatted.append(f"- {r['title']}: {r['content'][:300]} (source: {r['url']})")
    return "\n".join(formatted)


def calculator(expression: str) -> str:
    """
    A basic calculator tool. LLMs are unreliable at arithmetic, so giving
    them an explicit calculator tool (rather than trusting them to do math
    in their head) is standard practice in production agents.
    """
    # eval() is normally unsafe — here it's restricted to arithmetic only
    # by stripping anything that isn't a digit/operator, since this tool
    # is a portfolio demo, not something exposed to untrusted internet input.
    allowed_chars = set("0123456789+-*/(). ")
    if not set(expression) <= allowed_chars:
        return "Error: expression contains disallowed characters."
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error evaluating expression: {e}"


# JSON-schema tool descriptions, following the OpenAI/Groq function-calling format
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for current information. Use this whenever you "
                "need facts, current events, or anything you're not certain about."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a basic arithmetic expression, e.g. '12 * (4 + 7)'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A basic arithmetic expression using +, -, *, /, and parentheses.",
                    }
                },
                "required": ["expression"],
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "web_search": web_search,
    "calculator": calculator,
}
