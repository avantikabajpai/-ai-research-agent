# AI Research Agent

A full-stack autonomous research agent built on the ReAct (Reasoning + Acting) pattern, implemented in raw Python with no agent framework. Includes persistent vector memory, a FastAPI backend, and a React chat frontend.

## Overview

Most lightweight AI projects are a single LLM API call. This project implements the actual mechanics of an autonomous agent: the model decides which tool to call, the backend executes it, the result feeds back into the conversation, and this repeats until the model has enough to answer. Combined with semantic memory across sessions, a REST API, and a chat UI - the same architectural pattern used in production agentic systems like Claude Code and ChatGPT's browsing mode.

## Features

- Autonomous multi-step reasoning with tool use (web search, calculator)
- Long-term semantic memory via vector embeddings
- Short-term conversation memory within a session
- REST API with auto-generated interactive docs
- React chat interface
- Automated evaluation suite for regression testing
- Graceful handling of malformed tool calls and unanswerable questions
- Dockerized, deployable to a public URL

## Tech stack

| Layer | Technology |
|---|---|
| LLM | Llama-family model via Groq API |
| Agent logic | Raw Python (no LangChain) |
| Web search | Tavily API |
| Vector memory | ChromaDB |
| Backend | FastAPI, Uvicorn |
| Frontend | React, Vite |
| Containerization | Docker, Docker Compose |

## Architecture

frontend (React) --HTTP--> backend (FastAPI) --calls--> agent.py (ReAct loop)
                                                              |
                                                     tools.py (web_search, calculator)
                                                              |
                                                     memory.py (ChromaDB)

## Getting started

Backend:
cd backend
cp .env.example .env
pip install -r requirements.txt
python main.py

Frontend (second terminal):
cd frontend
npm install
npm run dev

Or together:
docker compose up --build

CLI only, no API/frontend:
cd backend && python run.py

Run evaluations:
cd backend && python eval.py

## Deployment

Backend deploys to any Dockerfile-supporting host (e.g. Render) with GROQ_API_KEY and TAVILY_API_KEY set as environment variables. Frontend deploys to any static host (e.g. Vercel) with VITE_API_URL pointing to the deployed backend.

## Key design decisions

- Raw Python over LangChain - every reasoning step is explicit, inspectable code
- Vector memory over a plain chat log - semantic search matches relevant past research even with different phrasing (RAG)
- max_iterations safety valve - prevents indefinite tool-call loops
- Multi-stage frontend Dockerfile - final image ships only compiled static files, not the Node build toolchain
- Separate error handling for malformed tool calls vs. tool execution failures - falls back to a plain-text response instead of crashing

## Limitations

- Conversation history is per-session only, not persisted across reloads
- Search results depend on index freshness - can lag for fast-moving data
- No auth/rate limiting - not intended for public unauthenticated deployment as-is
