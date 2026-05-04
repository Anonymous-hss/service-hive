"""
RAG ingest CLI.

Usage:
    python -m app.rag.ingest --path docs/
    python -m app.rag.ingest --path docs/ --chunk-size 512 --chunk-overlap 64

Reads markdown files, chunks them, embeds, and writes to the vector store.
"""
import argparse
import asyncio
import hashlib
from pathlib import Path
import re
import yaml
import chromadb
import google.generativeai as genai

from app.settings import settings

if settings.google_api_key:
    genai.configure(api_key=settings.google_api_key)


def chunk_sentences(text: str, max_chars: int = 512, overlap_sentences: int = 1) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current, current_len = [], [], 0
    for sentence in sentences:
        if current_len + len(sentence) > max_chars and current:
            chunks.append(" ".join(current))
            current = current[-overlap_sentences:]  # keep last N sentences as overlap
            current_len = sum(len(s) for s in current)
        current.append(sentence)
        current_len += len(sentence)
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_markdown(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """
    Split markdown text into overlapping chunks.
    Split on ## and ### headings, and sub-chunk if needed.
    """
    sections = re.split(r'\n(?=#{2,3} )', text)
    chunks = []
    for section in sections:
        if len(section) <= chunk_size:
            chunks.append(section.strip())
        else:
            chunks.extend(chunk_sentences(section, max_chars=chunk_size))
    return [c for c in chunks if c.strip()]


def extract_metadata(file_path: Path, text: str) -> tuple[dict, str]:
    """
    Extract metadata from a markdown file's frontmatter and return (metadata, body).
    """
    match = re.match(r'^---\n(.*?)\n---\n', text, re.DOTALL)
    metadata = {}
    body = text
    if match:
        try:
            metadata = yaml.safe_load(match.group(1)) or {}
        except Exception as e:
            # Frontmatter parse failed — skip metadata, use defaults
            import logging
            logging.getLogger(__name__).debug("Failed to parse frontmatter in %s: %s", file_path, e)
        body = text[match.end():]

    filtered_meta = {}
    if isinstance(metadata, dict):
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)):
                filtered_meta[k] = v
            elif isinstance(v, list):
                filtered_meta[k] = ",".join(str(item) for item in v)
            else:
                filtered_meta[k] = str(v)

    filtered_meta["source"] = file_path.name
    return filtered_meta, body


def make_chunk_id(file_name: str, chunk_index: int) -> str:
    raw = f"{file_name}::{chunk_index}"
    return "chunk_" + hashlib.sha256(raw.encode()).hexdigest()[:16]


from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type
from google.api_core.exceptions import ResourceExhausted

@retry(
    wait=wait_random_exponential(min=5, max=60),
    stop=stop_after_attempt(10),
    retry=retry_if_exception_type(ResourceExhausted),
    reraise=True
)
def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts using Google's gemini-embedding-001 model with retry."""
    result = genai.embed_content(
        model="models/gemini-embedding-001",
        content=texts,
        task_type="retrieval_document",
    )
    embeddings = result.get("embedding") or result.get("embeddings")
    if embeddings is None:
        raise ValueError("No embeddings returned by genai API")
    return embeddings


async def embed_in_batches(texts: list[str], batch_size: int = 50) -> list[list[float]]:
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        loop = asyncio.get_running_loop()
        batch_embeddings = await loop.run_in_executor(None, embed_texts, batch)
        embeddings.extend(batch_embeddings)
        if i + batch_size < len(texts):
            await asyncio.sleep(2)
    return embeddings


async def ingest_directory(docs_path: Path, chunk_size: int, chunk_overlap: int) -> None:
    """
    Walk docs_path, chunk and embed every .md file, upsert into vector store.
    """
    md_files = list(docs_path.rglob("*.md"))
    print(f"Found {len(md_files)} markdown files in {docs_path}")

    # Set up chroma
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    collection = client.get_or_create_collection(
        name="helix_docs",
        metadata={"hnsw:space": "cosine"},
    )

    ids = []
    embeddings = []
    documents = []
    metadatas = []

    for file_path in md_files:
        text = file_path.read_text(encoding="utf-8")
        metadata, body = extract_metadata(file_path, text)
        chunks = chunk_markdown(body, chunk_size, chunk_overlap)
        print(f"  {file_path.name}: {len(chunks)} chunks")

        for idx, chunk in enumerate(chunks):
            cid = make_chunk_id(file_path.name, idx)
            ids.append(cid)
            documents.append(chunk)
            metadatas.append(metadata)

    if documents:
        print(f"Embedding {len(documents)} total chunks...")
        chunk_embeddings = await embed_in_batches(documents)

        # Chroma upsert
        print("Upserting chunks to ChromaDB...")
        collection.upsert(
            ids=ids,
            embeddings=chunk_embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    print("Ingest complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest docs into the vector store")
    parser.add_argument("--path", type=Path, required=True, help="Directory containing .md files")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=64)
    args = parser.parse_args()

    asyncio.run(ingest_directory(args.path, args.chunk_size, args.chunk_overlap))


if __name__ == "__main__":
    main()
