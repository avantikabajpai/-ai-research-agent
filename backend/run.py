"""
run.py — the actual entry point you run from the terminal.

Loads .env FIRST, then imports agent.py. This ordering matters: agent.py and
tools.py read os.environ at import time (when the Groq/Tavily clients are
created), so the environment variables must already exist before those
imports happen.
"""

from dotenv import load_dotenv

load_dotenv()

from agent import run_agent  # noqa: E402  (import after load_dotenv on purpose)

if __name__ == "__main__":
    question = input("Ask the research agent something: ")
    answer = run_agent(question)
    print("\n=== FINAL ANSWER ===")
    print(answer)
