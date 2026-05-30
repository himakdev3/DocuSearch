import streamlit as st

from src.config.config_manager import config
from src.pipeline.document_pipeline import initialize_pipeline, process_documents
from src.utils.css import apply_custom_css
from src.utils.ui import search_interface




def run_streamlit_app() -> None:
    """Main app flow for Streamlit entrypoint."""
    _configure_page()

    _display_header()
    uploaded_files = _sidebar_configuration()
    _render_processing_controls(uploaded_files)

    pipeline = st.session_state.get("pipeline")
    if st.session_state.get("pipeline_ready", False) and pipeline is not None:
        search_interface(pipeline, config.default_top_k, config.min_similarity_threshold)
    else:
        st.info("Upload documents, click Process Documents, then search across documents.")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _configure_page() -> None:
    """Configure Streamlit page settings and apply global CSS."""
    st.set_page_config(
        page_title=config.app_title,
        page_icon=config.app_icon,
        layout=config.page_layout,
        initial_sidebar_state="expanded",
    )
    apply_custom_css()


def _render_processing_controls(uploaded_files) -> None:
    """Render sidebar actions for document processing."""

    with st.sidebar:
        st.markdown("---")
        if st.button("Process Documents", type="primary", use_container_width=True):
            if uploaded_files:
                if st.session_state.get("pipeline") is None:
                    with st.spinner("Preparing search engine..."):
                        st.session_state["pipeline"] = initialize_pipeline()
                process_documents(st.session_state["pipeline"], uploaded_files)
            else:
                st.warning("Please upload files first")



def _display_header() -> None:
    """Render the application header and introductory text."""
    st.markdown(
        f'<div class="main-header">{config.app_icon} {config.app_title}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">Upload documents and search them using natural language.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")


def _sidebar_configuration():
    """Render sidebar controls and return uploaded files."""
    with st.sidebar:
        st.header("Documents")
        return st.file_uploader(
            "Upload PDF, Word, Text, or PowerPoint documents",
            type=["pdf", "docx", "txt", "pptx"],
            accept_multiple_files=True,
            help="Upload one or more .pdf, .docx, .txt, or .pptx files",
        )
