from typing import List

import streamlit as st

from src.config.config_manager import config
from src.retrieval.rag_pipeline import RAGPipeline


@st.cache_resource(show_spinner="Loading search engine...")
def initialize_pipeline() -> RAGPipeline:
    """Initialize the RAG pipeline with Streamlit resource caching."""
    try:
        return RAGPipeline(
            embedding_model_name=config.embedding_model_name,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            use_gpu=config.use_gpu,
        )
    except Exception as exc:
        st.error(f"Failed to initialize RAG pipeline: {exc}")
        st.error("Check your configuration and model installation.")
        raise RuntimeError(f"Pipeline initialization failed: {exc}") from exc


def process_documents(pipeline: RAGPipeline, uploaded_files: List) -> bool:
    """Load files, create embeddings, and build the vector index."""
    if not uploaded_files:
        st.warning("Please upload PDF, Word, Text, or PowerPoint (.pptx) files to begin.")
        return False

    try:
        with st.spinner("Processing documents..."):
            progress_bar, status_text = _create_progress_widgets()

            if not _load_documents(pipeline, uploaded_files, progress_bar, status_text):
                _clear_progress_widgets(progress_bar, status_text)
                return False

            _generate_embeddings(pipeline, progress_bar, status_text)
            _build_index(pipeline, progress_bar, status_text)

            _finalize_success(progress_bar, status_text)
            return True
    except Exception as exc:
        st.error(f"Error processing documents: {exc}")
        return False


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _create_progress_widgets():
    """Create and return progress UI components."""
    return st.progress(0), st.empty()


def _load_documents(pipeline: RAGPipeline, uploaded_files: List, progress_bar, status_text) -> bool:
    """Load uploaded documents and validate extracted content."""
    status_text.text("Loading documents...")
    progress_bar.progress(20)
    num_chunks = pipeline.load_documents_from_files(uploaded_files)
    if num_chunks == 0:
        st.error("No text could be extracted from the files. Please check your uploads.")
        return False
    return True


def _generate_embeddings(pipeline: RAGPipeline, progress_bar, status_text) -> None:
    """Generate vector embeddings for loaded document chunks."""
    status_text.text("Generating embeddings...")
    progress_bar.progress(50)
    pipeline.generate_embeddings(batch_size=config.embedding_batch_size, show_progress=False)


def _build_index(pipeline: RAGPipeline, progress_bar, status_text) -> None:
    """Build the retrieval index from generated embeddings."""
    status_text.text("Building search index...")
    progress_bar.progress(80)
    pipeline.build_index()


def _clear_progress_widgets(progress_bar, status_text) -> None:
    """Clear progress widgets from the sidebar panel."""
    progress_bar.empty()
    status_text.empty()


def _finalize_success(progress_bar, status_text) -> None:
    """Finalize UI state after successful processing."""
    progress_bar.progress(100)
    _clear_progress_widgets(progress_bar, status_text)
    st.session_state["pipeline_ready"] = True
