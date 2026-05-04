"""
Unit tests for RAG retrieval.
"""
import pytest
from app.rag.ingest import chunk_markdown, make_chunk_id


@pytest.mark.asyncio
async def test_search_docs_returns_results_with_chunk_ids():
    """search_docs must return chunk IDs and scores in [0, 1]."""
    from app.agents.tools.search_docs import search_docs
    results = await search_docs("how to rotate a deploy key", k=3)
    # If vector store is empty (no docs ingested), skip gracefully
    if not results:
        pytest.skip("Vector store is empty — run ingest first for full test")
    assert len(results) > 0
    assert all(r.chunk_id for r in results)
    assert all(0.0 <= r.score <= 1.0 for r in results)


def test_chunker_produces_non_empty_chunks():
    """Chunker must not produce empty strings."""
    text = "# Header\n\nSome content.\n\n## Section 2\n\nMore content here."
    chunks = chunk_markdown(text, chunk_size=100, overlap=20)
    assert len(chunks) > 0
    assert all(c.strip() for c in chunks)


def test_chunk_ids_are_deterministic():
    """Stable chunk IDs: same input must produce the same ID."""
    id1 = make_chunk_id("deploy-keys.md", 0)
    id2 = make_chunk_id("deploy-keys.md", 0)
    id3 = make_chunk_id("deploy-keys.md", 1)
    assert id1 == id2, "Same input should produce the same chunk ID"
    assert id1 != id3, "Different chunk indices should produce different IDs"
    assert id1.startswith("chunk_"), "Chunk ID should be prefixed with 'chunk_'"


def test_chunker_respects_heading_boundaries():
    """Heading-aware chunking should split on ## boundaries."""
    text = "## Section A\n\nContent A here.\n\n## Section B\n\nContent B here."
    chunks = chunk_markdown(text, chunk_size=500, overlap=0)
    assert len(chunks) >= 2
    assert any("Section A" in c for c in chunks)
    assert any("Section B" in c for c in chunks)
