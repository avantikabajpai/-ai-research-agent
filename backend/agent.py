"""
agent.py — A ReAct-pattern autonomous agent, built from raw HTTP calls to Groq.

WHY NO LANGCHAIN?
LangChain would hide the actual mechanics (the Thought -> Action -> Observation
loop, how tool results get fed back to the model, how the model decides when
it's "done"). Building it raw means you can explain every line in an interview,
because you wrote every line.

HOW A ReAct AGENT WORKS (read this before the code):
1. We give the LLM a system prompt describing tools it can call and asking it
   to think step by step.
2. The LLM responds with either:
     a) a tool call (it wants more information), or
     b) a final answer (it has enough information).
3. If it's a tool call, WE (the Python code) actually run the tool, and feed
   the tool's output back into the conversation as a new message.
4. We loop back to step 2. This repeats until the LLM gives a final answer,
   or we hit a safety limit on iterations.

This is the exact loop every "agentic" AI product implements under the hood —
Claude Code, ChatGPT's browsing mode, AutoGPT, all of it. Understanding this
loop cold is the single highest-value thing you can say in an interview.
"""

import os
import json
from groq import Groq
from tools import TOOLS, TOOL_FUNCTIONS
from memory import store_memory, retrieve_relevant

client = Groq(api_key=os.environ["GROQ_API_KEY"])

MODEL = "openai/gpt-oss-120b"

SYSTEM_PROMPT = """You are a research assistant with access to tools.

When you need information you don't already have, call the appropriate tool.
Do not guess or make up facts — use the search tool to verify anything you're
unsure about.

Some questions cannot be answered by any tool, no matter how you search --
for example, exact future prices, events that haven't happened yet, or
anything inherently unpredictable. If a first search doesn't turn up a real
answer, do NOT keep retrying with reworded queries. Recognize the question
is unanswerable and say so directly as your final answer, explaining why.

Once you have enough information to fully answer the user's question, give a
clear, well-organized final answer. Do not call more tools than necessary.
"""

def run_agent(
    user_question: str,
    conversation_history: list[dict] | None = None,
    max_iterations: int = 6,
    verbose: bool = True,
) -> str:
    """
    Runs the ReAct loop until the model produces a final answer (no tool call)
    or we hit max_iterations (a safety valve against infinite loops — an agent
    that keeps calling tools forever is a real failure mode you should always
    guard against).
    """
    system_prompt = SYSTEM_PROMPT

    # Stage 2: pull in relevant past research before we even start reasoning.
    past_context = retrieve_relevant(user_question)
    if past_context:
        joined = "\n\n".join(past_context)
        system_prompt += (
            "\n\nYou have relevant memory from past research sessions below. "
            "If it clearly answers or relates to the current question, USE IT "
            "confidently instead of asking the user to clarify — for example, "
            "if memory shows a single country was just discussed, assume a "
            "vague follow-up like 'that country' refers to it. Only ask for "
            "clarification if memory has multiple equally-plausible matches "
            "or none at all. Still verify anything time-sensitive with a "
            f"fresh tool call rather than trusting stored data blindly:\n{joined}"
        )


    messages = [{"role": "system", "content": system_prompt}]

    if conversation_history:
        messages.extend(conversation_history)

    messages.append({"role": "user", "content": user_question})


    for iteration in range(1, max_iterations + 1):
        if verbose:
            print(f"\n--- Iteration {iteration} ---")

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )
        except Exception as e:
            if verbose:
                print(f"API call failed ({e}); retrying once without tools.")
            fallback = client.chat.completions.create(
                model=MODEL,
                messages=messages + [
                    {
                        "role": "user",
                        "content": (
                            "(A tool call failed validation. Please answer "
                            "using only what you already know, and say so if "
                            "you're not certain.)"
                        ),
                    }
                ],
                tool_choice="none",
            )
            store_message = fallback.choices[0].message.content
            store_memory(user_question, store_message)
            return store_message


        message = response.choices[0].message

        # Case 1: the model wants to call one or more tools
        if message.tool_calls:
            # We must append the assistant's tool-call message to history
            # before appending tool results, or the API will reject the
            # conversation as malformed.
            messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in message.tool_calls
                    ],
                }
            )

            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                if verbose:
                    print(f"Thought -> calling tool: {tool_name}({tool_args})")

                tool_fn = TOOL_FUNCTIONS.get(tool_name)
                if tool_fn is None:
                    result = f"Error: unknown tool '{tool_name}'"
                else:
                    try:
                        result = tool_fn(**tool_args)
                    except Exception as e:
                        # Tools fail in the real world (bad args, network
                        # errors, rate limits). Feeding the error back to the
                        # model lets it adapt instead of crashing the whole run.
                        result = f"Error running {tool_name}: {e}"

                if verbose:
                    print(f"Observation -> {str(result)[:300]}")

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result),
                    }
                )

            # Loop again: send the tool results back to the model
            continue

        # Case 2: no tool call -> the model is giving its final answer
        if verbose:
            print(f"Final answer reached after {iteration} iteration(s).")
        store_memory(user_question, message.content)
        return message.content

    return (
        "Agent stopped: reached max_iterations without a final answer. "
        "This usually means the task needs breaking into smaller steps."
    )


if __name__ == "__main__":
    question = input("Ask the research agent something: ")
    answer = run_agent(question)
    print("\n=== FINAL ANSWER ===")
    print(answer)
