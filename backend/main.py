"""
main.py — Stage 3: wraps the agent in a FastAPI HTTP server.

This is the piece that turns "a script I run in a terminal" into "a
full-stack project" — the React frontend (Stage 4) talks to this API,
not directly to agent.py.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from agent import run_agent  # noqa: E402  (must load .env first)

app = FastAPI(title="AI Research Agent API")

# CORS: without this, a browser-based React app running on a different
# origin (e.g. localhost:5173) would be blocked from calling this API by
# the browser's same-origin policy. In production you'd restrict this to
# your actual frontend's domain instead of "*".
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str
    history: list[dict] = []


class AskResponse(BaseModel):
    answer: str


@app.get("/health")
def health():
    """Simple endpoint to confirm the API is up — useful for deployment
    platforms that ping this to check the service is alive."""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """
    Runs the full ReAct agent loop for a single question and returns the
    final answer. verbose=False here because the Thought/Action/Observation
    prints are meant for your terminal, not an HTTP response.
    """
    answer = run_agent(request.question, conversation_history=request.history, verbose=False)
    return AskResponse(answer=answer)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
