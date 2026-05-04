"""
search_docs tool — used by KnowledgeAgent.

Queries the vector store for relevant documentation chunks.
Returns chunk IDs, scores, and content so the agent can cite sources.
"""
from dataclasses import dataclass
import chromadb
import google.generativeai as genai
import asyncio
from app.settings import settings
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type
from google.api_core.exceptions import ResourceExhausted

if settings.google_api_key:
    genai.configure(api_key=settings.google_api_key)


@dataclass
class DocChunk:
    chunk_id: str
    score: float
    content: str
    metadata: dict  # e.g. {"product_area": "security", "source": "deploy-keys.md"}


@retry(
    wait=wait_random_exponential(min=5, max=60),
    stop=stop_after_attempt(10),
    retry=retry_if_exception_type(ResourceExhausted),
    reraise=True
)
def embed_query(query: str) -> list[float]:
    """Embed the query using Google's gemini-embedding-001 model with retry."""
    result = genai.embed_content(
        model="models/gemini-embedding-001",
        content=query,
        task_type="retrieval_query",
    )
    embedding = result.get("embedding") or result.get("embeddings")
    if embedding is None:
        raise ValueError("No embeddings returned by genai API")
    return embedding


async def search_docs(query: str, k: int = 5, product_area: str | None = None) -> list[DocChunk]:
    """
    Search the vector store for top-k relevant chunks.
    """
    loop = asyncio.get_running_loop()
    query_embedding = await loop.run_in_executor(None, embed_query, query)

    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    collection = client.get_or_create_collection(
        name="helix_docs",
        metadata={"hnsw:space": "cosine"},
    )

    where = {"product_area": product_area} if product_area else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where=where,
    )

    chunks = []
    if results and "ids" in results and len(results["ids"]) > 0:
        for cid, distance, doc, meta in zip(
            results["ids"][0],
            results["distances"][0],
            results["documents"][0],
            results["metadatas"][0],
        ):
            chunks.append(DocChunk(
                chunk_id=cid,
                score=round(1.0 - distance, 4),
                content=doc,
                metadata=meta,
            ))

    return sorted(chunks, key=lambda c: c.score, reverse=True)
