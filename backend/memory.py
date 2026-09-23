"""
memory.py — Stage 2: gives the agent persistent memory across runs using a
vector database (ChromaDB).

WHY VECTOR MEMORY?
A plain dict/list of past Q&A pairs only helps if you search for the exact
same words again. Vector memory converts text into embeddings (numbers that
capture *meaning*), so a new question like "How's Bitcoin doing?" can match
a stored memory about "current price of Bitcoin" even though the wording is
different. This semantic-similarity retrieval is the core idea behind every
production RAG (Retrieval-Augmented Generation) system.

HOW IT'S USED HERE:
1. Before answering, we embed the new question and search ChromaDB for the
   most similar past Q&A pairs.
2. If anything relevant turns up, we inject it into the system prompt as
   background context — the agent can build on past research instead of
   starting from zero every time.
3. After answering, we store the new Q&A pair so future questions can draw
   on it too.
"""

import chromadb

# PersistentClient writes to disk (./chroma_db) so memory survives between
# runs, not just within a single process — that's the whole point of Stage 2.
_client = chromadb.PersistentClient(path="./chroma_db")

_collection = _client.get_or_create_collection(
    name="research_memory",
    metadata={"hnsw:space": "cosine"},  # cosine similarity is standard for text embeddings
)


def store_memory(question: str, answer: str) -> None:
    """Saves a completed Q&A pair into the vector store."""
    existing_count = _collection.count()
    _collection.add(
        documents=[f"Q: {question}\nA: {answer}"],
        ids=[f"memory-{existing_count}"],
        metadatas=[{"question": question}],
    )


def retrieve_relevant(query: str, n_results: int = 3) -> list[str]:
    """
    Returns up to n_results past Q&A pairs whose meaning is closest to the
    new query. Returns an empty list if memory is empty or nothing is
    relevant enough — this keeps the agent's first-ever run working exactly
    like Stage 1, with no crashes on a fresh, empty database.
    """
    if _collection.count() == 0:
        return []

    results = _collection.query(
        query_texts=[query],
        n_results=min(n_results, _collection.count()),
    )
    documents = results.get("documents", [[]])[0]
    return documents


if __name__ == "__main__":
    # Quick manual test: run `python memory.py` to sanity-check storage/retrieval.
    store_memory("What is the capital of France?", "Paris.")
    store_memory("What is the tallest mountain?", "Mount Everest, at 8,849m.")
    print(retrieve_relevant("Tell me about France's capital city"))
