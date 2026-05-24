import numpy as np
import pytest

from src.retrieval.rag_pipeline import DocumentChunk, RAGPipeline


def _pipeline_for_chunk_tests(chunk_size: int = 60, overlap: int = 10) -> RAGPipeline:
    """Create a lightweight pipeline instance without loading ML models."""
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.chunk_size = chunk_size
    pipeline.chunk_overlap = overlap
    return pipeline


def test_create_chunks_returns_single_chunk_for_short_text() -> None:
    pipeline = _pipeline_for_chunk_tests(chunk_size=80, overlap=10)
    text = "This is a short paragraph that should not be split."

    chunks = pipeline._create_chunks(text)

    assert len(chunks) == 1
    assert chunks[0] == text


def test_create_chunks_splits_long_text_without_empty_chunks() -> None:
    pipeline = _pipeline_for_chunk_tests(chunk_size=70, overlap=15)
    text = (
        "Machine learning enables computers to improve from data. "
        "Document retrieval systems rely on embedding vectors for semantic similarity. "
        "Chunking strategy affects both retrieval quality and explainability."
    )

    chunks = pipeline._create_chunks(text)

    assert len(chunks) >= 2
    assert all(chunk.strip() for chunk in chunks)
    assert all(len(chunk) <= pipeline.chunk_size for chunk in chunks)


def test_create_chunks_preserves_word_boundaries_at_chunk_start() -> None:
    pipeline = _pipeline_for_chunk_tests(chunk_size=55, overlap=12)
    text = (
        "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu. "
        "Nu xi omicron pi rho sigma tau upsilon phi chi psi omega."
    )

    chunks = pipeline._create_chunks(text)

    assert len(chunks) >= 2
    # Follow-up chunks should begin with a whole token, not whitespace.
    for chunk in chunks[1:]:
        assert chunk[0].isalnum()


class _FakeEmbeddingModel:
    def encode(self, *_args, **_kwargs):
        return np.array([[1.0, 0.0]], dtype="float32")


class _FakeIndex:
    def __init__(self, scores, indices):
        self._scores = scores
        self._indices = indices

    def search(self, _query_embedding, _top_k):
        return self._scores, self._indices


def _pipeline_for_retrieve() -> RAGPipeline:
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.is_indexed = True
    pipeline.embedding_model = _FakeEmbeddingModel()
    pipeline.documents = [
        DocumentChunk(text="alpha", doc_name="a.pdf", page_num=1, chunk_id=0),
        DocumentChunk(text="beta", doc_name="b.pdf", page_num=1, chunk_id=1),
    ]
    pipeline.index = _FakeIndex(
        scores=np.array([[0.91, 0.22]], dtype="float32"),
        indices=np.array([[0, 1]], dtype="int64"),
    )
    return pipeline


def test_retrieve_applies_min_score_threshold() -> None:
    pipeline = _pipeline_for_retrieve()

    results = pipeline.retrieve("query", top_k=2, min_score=0.5)

    assert len(results) == 1
    assert results[0][0].doc_name == "a.pdf"
    assert results[0][1] == pytest.approx(0.91, rel=1e-4)


def test_retrieve_rejects_invalid_min_score() -> None:
    pipeline = _pipeline_for_retrieve()

    with pytest.raises(ValueError, match="min_score must be between 0 and 1"):
        pipeline.retrieve("query", top_k=2, min_score=1.5)


def test_build_index_requires_embeddings_matrix() -> None:
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.documents = [DocumentChunk(text="alpha", doc_name="a.pdf", page_num=1, chunk_id=0)]
    pipeline.embeddings_matrix = None

    with pytest.raises(ValueError, match="Generate embeddings before building index"):
        pipeline.build_index()
