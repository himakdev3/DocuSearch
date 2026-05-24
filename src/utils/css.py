"""
Custom CSS styling for Streamlit application.

This module contains all CSS styles used in the RAG Document Assistant UI.
"""


def apply_custom_css():
    """
    Apply custom CSS to the Streamlit app.

    This function should be called once at the app initialization.
    """
    import streamlit as st
    st.markdown(_get_custom_css(), unsafe_allow_html=True)


def _get_custom_css() -> str:
    """
    Return custom CSS for the Streamlit application.
    
    Returns:
        CSS string to be injected into the app
    """
    return """
    <style>
    :root {
        --brand-blue: #1E88E5;
        --brand-blue-deep: #1565C0;
        --brand-blue-soft: #64B5F6;
        --brand-blue-glow: rgba(30, 136, 229, 0.28);
    }
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: var(--brand-blue);
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 0 10px 30px rgba(30, 136, 229, 0.18);
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid var(--brand-blue);
    }
    .citation-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 3px solid #4CAF50;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
        border: 1px solid rgba(21, 101, 192, 0.22);
        border-radius: 999px;
        background: linear-gradient(180deg, #4DA3FF 0%, #1E88E5 55%, #1565C0 100%);
        color: #ffffff;
        font-weight: 600;
        box-shadow: 0 10px 24px var(--brand-blue-glow), inset 0 1px 0 rgba(255, 255, 255, 0.28);
        transition: transform 0.16s ease, box-shadow 0.16s ease, filter 0.16s ease;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        filter: brightness(1.03);
        box-shadow: 0 14px 30px rgba(21, 101, 192, 0.34), inset 0 1px 0 rgba(255, 255, 255, 0.34);
        color: #ffffff;
        border-color: rgba(21, 101, 192, 0.28);
    }
    .stButton>button:focus,
    .stButton>button:focus-visible {
        outline: none;
        color: #ffffff;
        box-shadow: 0 0 0 0.18rem rgba(100, 181, 246, 0.35), 0 14px 30px rgba(21, 101, 192, 0.34);
    }
    .stButton>button[kind="secondary"] {
        background: linear-gradient(180deg, #4DA3FF 0%, #1E88E5 55%, #1565C0 100%);
        color: #ffffff;
    }
    div[data-testid="stProgressBar"] > div > div {
        background: linear-gradient(90deg, #90CAF9 0%, #42A5F5 35%, #1E88E5 70%, #1565C0 100%);
        border-radius: 999px;
        box-shadow: 0 8px 18px rgba(30, 136, 229, 0.24);
    }
    div[data-testid="stProgressBar"] > div {
        background: linear-gradient(180deg, rgba(227, 242, 253, 0.95) 0%, rgba(207, 232, 255, 0.95) 100%);
        border-radius: 999px;
    }
    .info-badge {
        background-color: #E3F2FD;
        color: #1976D2;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.875rem;
        font-weight: 500;
        display: inline-block;
        margin: 0.25rem;
    }
    .success-badge {
        background-color: #E8F5E9;
        color: #388E3C;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.875rem;
        font-weight: 500;
        display: inline-block;
        margin: 0.25rem;
    }
    .warning-badge {
        background-color: #FFF3E0;
        color: #F57C00;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.875rem;
        font-weight: 500;
        display: inline-block;
        margin: 0.25rem;
    }
    .summary-metrics-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.85rem;
        margin: 0.75rem 0 0.35rem;
    }
    .summary-metric-card {
        background: linear-gradient(180deg, rgba(227, 242, 253, 0.96) 0%, rgba(207, 232, 255, 0.96) 100%);
        border: 1px solid rgba(30, 136, 229, 0.14);
        border-radius: 1rem;
        padding: 0.95rem 1rem;
        box-shadow: 0 10px 22px rgba(30, 136, 229, 0.12);
    }
    .summary-metric-label {
        font-size: 1rem;
        font-weight: 700;
        line-height: 1.2;
        color: #0D47A1;
        margin-bottom: 0.35rem;
    }
    .summary-metric-value {
        font-size: 0.95rem;
        font-weight: 500;
        color: #335c85;
        line-height: 1.35;
    }
    .pages-reference-card {
        margin-top: 0.8rem;
        padding: 1rem 1.05rem;
        border-radius: 1rem;
        border: 1px solid rgba(30, 136, 229, 0.14);
        background: linear-gradient(180deg, rgba(245, 250, 255, 0.98) 0%, rgba(233, 244, 255, 0.98) 100%);
        box-shadow: 0 10px 22px rgba(30, 136, 229, 0.08);
    }
    .pages-reference-title {
        font-size: 1rem;
        font-weight: 700;
        color: #0D47A1;
        margin-bottom: 0.55rem;
    }
    .pages-reference-row {
        font-size: 0.95rem;
        line-height: 1.45;
        color: #335c85;
        padding: 0.28rem 0;
        border-top: 1px solid rgba(30, 136, 229, 0.08);
    }
    .pages-reference-row:first-of-type {
        border-top: none;
        padding-top: 0;
    }
    .pages-reference-doc {
        font-weight: 600;
        color: #174a88;
    }
    @media (max-width: 900px) {
        .summary-metrics-grid {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """
