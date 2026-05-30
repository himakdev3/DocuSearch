import streamlit as st
import pandas as pd
import json
import re
import io
from html import escape
from datetime import datetime


_ENCODING_FIXES = {
    "â€™": "'",
    "â€˜": "'",
    "â€œ": '"',
    "â€": '"',
    "â€“": "-",
    "â€”": "-",
    "â€¦": "...",
    "LL Ms": "LLMs",
    "incontext": "in-context",
}

_MIN_FINAL_ANSWER_CHARS = 50


def _score_color(score: float) -> str:
    if score >= 0.55:
        return "🟢"
    if score >= 0.40:
        return "🟡"
    return "🔴"


def _score_band(score: float) -> str:
    if score >= 0.55:
        return "high"
    if score >= 0.40:
        return "medium"
    return "low"


def _display_page_num(chunk) -> int:
    """Return user-facing page number, preferring printed page metadata when available."""
    printed = chunk.metadata.get("printed_page_num")
    try:
        printed_int = int(printed)
        if printed_int > 0:
            return printed_int
    except (TypeError, ValueError):
        pass
    return chunk.page_num


def _display_page_text(chunk) -> str:
    """Return user-facing page label text."""
    shown = _display_page_num(chunk)
    return str(shown)


def search_interface(pipeline, top_k, min_similarity_threshold: float = 0.0):
    """Render search controls, results, and export actions."""
    query = st.text_input(
        "What would you like to search?",
        placeholder="e.g. Find condional statements from the documents?",
        key="search_query",
    )
    if st.button("Search", key="search_button", type="primary") and query:
        with st.spinner("Searching your documents…"):
            try:
                retrieved_chunks = pipeline.retrieve(
                    query,
                    top_k=top_k,
                    min_score=min_similarity_threshold,
                )
                unique = _deduplicate_results(retrieved_chunks)
                st.session_state["last_search_query"] = query
                st.session_state["last_search_results"] = unique
                st.session_state["last_search_ran"] = True

            except Exception as e:
                st.session_state.pop("last_search_results", None)
                st.session_state.pop("last_search_ran", None)
                st.error(f"Error during search: {e}")
                return

    if not query:
        return

    unique = st.session_state.get("last_search_results")
    last_query = st.session_state.get("last_search_query")
    search_ran = st.session_state.get("last_search_ran", False)

    if not search_ran or last_query != query:
        return

    if not unique:
        st.info("No results found. Try rephrasing your question.")
        return

    _render_final_answer_panel(query, unique)
    _render_citation_summary_table(unique)
    _render_results(unique)
    _render_formatted_results_table(query, unique)
    export_rows = _build_export_rows(query, unique)
    _render_export_actions(export_rows)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _deduplicate_results(retrieved_chunks):
    """Keep only the highest-scored chunk per (document, page)."""
    best: dict = {}
    for chunk, score in retrieved_chunks:
        key = (chunk.doc_name, chunk.page_num)
        if key not in best or score > best[key][1]:
            best[key] = (chunk, score)
    return sorted(best.values(), key=lambda x: x[1], reverse=True)


def _sentence_safe_excerpt(text: str) -> str:
    """Return a cleaner excerpt that avoids starting/ending with broken sentence fragments."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    # Remove known OCR/book-preview noise such as truncated short links.
    cleaned = re.sub(r"More books:\s*https?://[^\s]*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"https?://[A-Za-z0-9.-]*\.[A-Za-z]?\b", "", cleaned)
    cleaned = re.sub(r"([A-Z]{2,})([A-Z][a-z])", r"\1 \2", cleaned)
    cleaned = re.sub(
        r"^(?:\s*[A-Z][A-Z0-9&'()/:;,-]{2,}(?:\s+[A-Z][A-Z0-9&'()/:;,-]{2,}){3,18})(?=\s+[A-Z][a-z]|\s+[A-Z]{2,}\b)",
        "",
        cleaned,
    )
    cleaned = re.sub(r"\bby\s+[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3}(?:\s*,\s*[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3}){1,5}", "", cleaned)
    for bad, good in _ENCODING_FIXES.items():
        cleaned = cleaned.replace(bad, good)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return cleaned

    excerpt = cleaned
    trimmed_prefix = False
    trimmed_suffix = False

    # If chunk begins mid-sentence, skip to the next likely sentence boundary.
    if excerpt and excerpt[0].islower():
        lead_boundary = re.search(r"[.!?]\s+", excerpt[:220])
        if lead_boundary:
            excerpt = excerpt[lead_boundary.end():].lstrip()
            trimmed_prefix = True

    # If chunk ends mid-sentence, trim to the last sentence terminator.
    if excerpt and excerpt[-1] not in ".!?":
        tail_boundary = max(excerpt.rfind("."), excerpt.rfind("!"), excerpt.rfind("?"))
        if tail_boundary > max(40, len(excerpt) // 2):
            excerpt = excerpt[:tail_boundary + 1].rstrip()
            trimmed_suffix = True

    if not excerpt:
        return cleaned
    if trimmed_prefix:
        excerpt = f"... {excerpt}"
    if trimmed_suffix:
        excerpt = f"{excerpt} ..."

    return excerpt


def _split_sentences(text: str) -> list:
    """Split text into sentence-like segments."""
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p and p.strip()]


def _normalize_summary_sentence(sentence: str) -> str:
    """Clean a sentence for summary display by removing discourse noise and OCR artifacts."""
    normalized = re.sub(r"\s+", " ", sentence).strip(" .,")
    normalized = re.sub(
        r"^(moreover|thus|therefore|however|now|also|furthermore)\s*,?\s+",
        "",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(r"\bC\s+HAPTER\s+\d+\b", "", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\b(introduction|structure|objectives)\b\s*", "", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\s+", " ", normalized).strip(" .,")
    if normalized and normalized[-1] not in ".!?":
        normalized += "."
    return normalized


def _is_summary_candidate(sentence: str) -> bool:
    """Reject noisy, title-like, or OCR-fragment sentences for the final answer."""
    normalized = _normalize_summary_sentence(sentence)
    if not normalized:
        return False

    lowered = normalized.lower()
    if "more books" in lowered or "https://" in lowered or "http://" in lowered:
        return False

    banned_fragments = {
        "concept of exception handling",
        "python introduction",
        "similarities between",
        "differences between",
        "control flow of python exception hierarchy",
    }
    if any(fragment in lowered for fragment in banned_fragments):
        return False

    words = re.findall(r"[A-Za-z0-9']+", normalized)
    if len(words) < 8:
        return False

    # Reject table-of-contents or keyword-list style fragments.
    if normalized.count(" ") > 10 and normalized.count(",") == 0 and normalized.count(".") <= 1:
        title_case_ratio = sum(1 for word in words if word[:1].isupper()) / max(len(words), 1)
        if title_case_ratio > 0.45:
            return False

    if len(normalized) < 50:
        return False

    if lowered.startswith("users will be able") or lowered.startswith("the user will see"):
        return False

    return True


def _query_terms(query: str) -> set:
    """Extract meaningful query tokens for scoring sentence relevance."""
    tokens = re.findall(r"[A-Za-z0-9]+", query.lower())
    return {t for t in tokens if len(t) >= 3}


def _detect_query_intent(query: str) -> str:
    """Infer a broad answer intent from query phrasing.

    Returns one of: list, explanatory, factoid, general.
    """
    q = (query or "").strip().lower()
    if not q:
        return "general"

    # Generic intent cues based on common question framing.
    if re.search(r"\b(list|types|kinds|categories|examples|different)\b", q):
        return "list"
    if re.search(r"\b(how|why|explain|describe|difference|compare)\b", q):
        return "explanatory"
    if re.search(r"\b(what|which|who|when|where)\b", q):
        return "factoid"
    return "general"


def _is_sentence_complete(sentence: str) -> bool:
    """Heuristic to reject visibly truncated answer sentences."""
    s = (sentence or "").strip()
    if len(s) < _MIN_FINAL_ANSWER_CHARS:
        return False
    return s.endswith((".", "!", "?"))


def _evidence_quality_score(
    avg_relevance: float,
    query_term_coverage: float,
    complete_sentence_ratio: float,
    source_diversity_ratio: float,
) -> float:
    """Compute an evidence-calibrated confidence score in [0, 1]."""
    quality = (
        (0.45 * avg_relevance)
        + (0.30 * query_term_coverage)
        + (0.15 * complete_sentence_ratio)
        + (0.10 * source_diversity_ratio)
    )
    return max(0.0, min(1.0, quality))


def _best_sentence_for_query(text: str, terms: set) -> tuple[str, int]:
    """Pick the best query-aligned sentence and return (sentence, overlap_count)."""
    sentences = [sentence for sentence in _split_sentences(text) if _is_summary_candidate(sentence)]
    if not sentences:
        return "", 0

    def sentence_score(sentence: str) -> tuple:
        normalized = _normalize_summary_sentence(sentence)
        lowered = normalized.lower()
        words = set(re.findall(r"[A-Za-z0-9]+", lowered))
        overlap = len(words & terms) if terms else 0
        coverage = overlap / max(len(terms), 1) if terms else 0.0
        return overlap, coverage, len(normalized)

    best = max(sentences, key=sentence_score)
    best_overlap = sentence_score(best)[0]
    return _normalize_summary_sentence(best), best_overlap


def _confidence_label(score: float) -> str:
    """Map average relevance score to a user-friendly confidence label."""
    if score >= 0.65:
        return "High"
    if score >= 0.45:
        return "Medium"
    return "Low"


def _two_line_key_excerpt(text: str) -> str:
    """Build a short, readable key excerpt (roughly two lines) from top evidence."""
    excerpt = _sentence_safe_excerpt(text).strip()
    if not excerpt:
        return ""

    sentences = _split_sentences(excerpt)
    if len(sentences) >= 2:
        key_excerpt = " ".join(sentences[:2]).strip()
    else:
        key_excerpt = excerpt

    key_excerpt = re.sub(r"\s+", " ", key_excerpt).strip()
    if len(key_excerpt) > 260:
        key_excerpt = key_excerpt[:257].rstrip() + "..."
    return key_excerpt


def _pages_by_document(unique) -> dict:
    """Build a mapping of document name to sorted unique pages."""
    pages: dict = {}
    for chunk, _ in unique:
        pages.setdefault(chunk.doc_name, {})[chunk.page_num] = _display_page_text(chunk)
    return {
        doc: [label for _, label in sorted(page_map.items(), key=lambda item: item[0])]
        for doc, page_map in pages.items()
    }


def _render_summary_metrics(panel: dict) -> None:
    """Render final answer metrics in custom blue cards."""
    metrics_markup = f"""
    <div class="summary-metrics-grid">
        <div class="summary-metric-card">
            <div class="summary-metric-label">Confidence</div>
            <div class="summary-metric-value">{panel['confidence_label']} ({panel['confidence_score']:.0%})</div>
        </div>
        <div class="summary-metric-card">
            <div class="summary-metric-label">Source Coverage</div>
            <div class="summary-metric-value">{panel['doc_count']} docs / {panel['page_count']} pages</div>
        </div>
        <div class="summary-metric-card">
            <div class="summary-metric-label">Displayed Results</div>
            <div class="summary-metric-value">{panel['displayed_results_count']}</div>
        </div>
    </div>
    """
    st.markdown(metrics_markup, unsafe_allow_html=True)


def _render_pages_reference(pages_by_doc: dict) -> None:
    """Render pages-by-document as a styled reference card."""
    if not pages_by_doc:
        return

    rows = "".join(
        (
            '<div class="pages-reference-row">'
            f'<span class="pages-reference-doc">{doc}</span>: '
            f"pages {', '.join(str(page) for page in pages)}"
            '</div>'
        )
        for doc, pages in pages_by_doc.items()
    )
    markup = f"""
    <div class="pages-reference-card">
        <div class="pages-reference-title">Pages by document</div>
        {rows}
    </div>
    """
    st.markdown(markup, unsafe_allow_html=True)


def _render_citation_summary_table(unique) -> None:
    """Render a compact citation-first summary table in a styled section."""
    if not unique:
        return

    rows = []
    for chunk, score in unique:
        rows.append(
            {
                "source": chunk.doc_name,
                "page": _display_page_text(chunk),
                "score": score,
            }
        )

    rows = sorted(rows, key=lambda row: row["score"], reverse=True)
    table_rows = "".join(
        (
            "<tr>"
            f"<td>{escape(row['source'])}</td>"
            f"<td>{row['page']}</td>"
            f"<td><span class='citation-score-badge citation-score-{_score_band(row['score'])}'>{row['score']:.0%}</span></td>"
            "</tr>"
        )
        for row in rows
    )

    table_markup = f"""
    <div class="citation-summary-card">
        <div class="citation-summary-title">Citation Summary</div>
        <div class="citation-summary-legend">
            <span class="citation-legend-item"><span class="citation-score-badge citation-score-high">55%+</span> High</span>
            <span class="citation-legend-item"><span class="citation-score-badge citation-score-medium">40-55%</span> Medium</span>
            <span class="citation-legend-item"><span class="citation-score-badge citation-score-low">Below 40%</span> Low</span>
        </div>
        <table class="citation-summary-table">
            <thead>
                <tr>
                    <th>Source Name</th>
                    <th>Citation Page Number</th>
                    <th>Relevance score</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
    </div>
    """
    st.markdown(table_markup, unsafe_allow_html=True)


def _build_final_answer(query: str, unique) -> dict:
    """Create an answer summary plus confidence and source coverage details."""
    top_items = unique[:5]
    intent = _detect_query_intent(query)
    terms = _query_terms(query)
    selected_sentences = []
    evidence_overlaps = []
    matched_terms = set()
    complete_count = 0

    for chunk, _ in top_items:
        excerpt = _sentence_safe_excerpt(chunk.text)
        sentence, overlap = _best_sentence_for_query(excerpt, terms)
        if sentence and overlap > 0 and sentence not in selected_sentences:
            selected_sentences.append(sentence)
            evidence_overlaps.append(overlap)
            if _is_sentence_complete(sentence):
                complete_count += 1

            if terms:
                sentence_terms = set(re.findall(r"[A-Za-z0-9]+", sentence.lower()))
                matched_terms |= (sentence_terms & terms)

    avg_top_score = sum(score for _, score in top_items) / max(len(top_items), 1)
    overlap_ready = bool(evidence_overlaps) and (max(evidence_overlaps) >= 1)
    query_term_coverage = (len(matched_terms) / len(terms)) if terms else 0.0
    complete_sentence_ratio = (complete_count / len(selected_sentences)) if selected_sentences else 0.0
    unique_docs = {chunk.doc_name for chunk, _ in unique}
    source_diversity_ratio = min(1.0, len(unique_docs) / max(len(top_items), 1))
    confidence_score = _evidence_quality_score(
        avg_relevance=avg_top_score,
        query_term_coverage=query_term_coverage,
        complete_sentence_ratio=complete_sentence_ratio,
        source_diversity_ratio=source_diversity_ratio,
    )

    meets_intent_quality = False
    if intent == "list":
        meets_intent_quality = (
            len(selected_sentences) >= 2
            and query_term_coverage >= 0.25
            and complete_sentence_ratio >= 0.5
        )
    elif intent == "explanatory":
        meets_intent_quality = (
            len(selected_sentences) >= 1
            and query_term_coverage >= 0.15
            and complete_sentence_ratio >= 0.5
        )
    else:
        meets_intent_quality = (
            len(selected_sentences) >= 1
            and query_term_coverage >= 0.12
            and complete_sentence_ratio >= 0.5
        )

    if selected_sentences and overlap_ready and confidence_score >= 0.40 and meets_intent_quality:
        summary = " ".join(selected_sentences[:2])
        summary_is_generated = True
    else:
        best_locations = ", ".join(f"{chunk.doc_name} p.{_display_page_text(chunk)}" for chunk, _ in top_items[:3])
        summary = (
            "Most relevant passages were found in these locations: "
            f"{best_locations}. "
            "Use the cited snippets below to verify where the answer appears."
        )
        summary_is_generated = False

    if len(summary) < _MIN_FINAL_ANSWER_CHARS:
        best_locations = ", ".join(f"{chunk.doc_name} p.{_display_page_text(chunk)}" for chunk, _ in top_items[:3])
        summary = (
            "Top matches were too short or weakly aligned with your question. "
            f"Review these passages: {best_locations}."
        )
        summary_is_generated = False

    if len(summary) > 420:
        summary = summary[:417].rstrip() + "..."

    unique_pages = {(chunk.doc_name, chunk.page_num) for chunk, _ in unique}
    top_sources = [f"{chunk.doc_name} p.{_display_page_text(chunk)}" for chunk, _ in top_items]
    pages_by_doc = _pages_by_document(unique)
    key_excerpt = _two_line_key_excerpt(top_items[0][0].text) if top_items else ""

    return {
        "summary": summary,
        "summary_is_generated": summary_is_generated,
        "key_excerpt": key_excerpt,
        "confidence_score": confidence_score,
        "confidence_label": _confidence_label(confidence_score),
        "doc_count": len(unique_docs),
        "page_count": len(unique_pages),
        "summary_evidence_count": len(top_items),
        "displayed_results_count": len(unique),
        "top_sources": top_sources,
        "pages_by_doc": pages_by_doc,
    }


def _render_final_answer_panel(query: str, unique) -> None:
    """Render a concise synthesized answer with confidence and coverage."""
    panel = _build_final_answer(query, unique)

    st.markdown("### Search Summary")
    if panel["summary_is_generated"]:
        st.info(panel["summary"])
    else:
        st.warning(panel["summary"])

    _render_summary_metrics(panel)

    _render_pages_reference(panel["pages_by_doc"])

    st.divider()


def _render_results(unique) -> None:
    """Render unique retrieval results in expandable cards."""
    max_score = max(s for _, s in unique)
    if max_score < 0.3:
        st.warning("Low match confidence — results may not be closely related to your question.")

    st.markdown(f"**{len(unique)} unique page result{'s' if len(unique) != 1 else ''} found**")
    st.caption("Results are deduplicated to one best match per document page.")

    for idx, (chunk, score) in enumerate(unique, 1):
        is_ocr = chunk.metadata.get("source") == "image_ocr"
        page_text = _display_page_text(chunk)
        label = f"{_score_color(score)} **{chunk.doc_name}** · Page {page_text}"
        if is_ocr:
            label += " · 🖼️ Image OCR"

        with st.expander(label, expanded=(idx == 1)):
            st.caption(f"Relevance score: {score:.0%}")
            preview_image = chunk.metadata.get("preview_image")
            preview_kind = chunk.metadata.get("preview_kind")
            file_type = (chunk.metadata.get("file_type") or "").lower()

            if _is_valid_preview_image(preview_image) and preview_kind == "pdf_page":
                text_col, preview_col = st.columns([1.35, 1])
                with text_col:
                    st.markdown(_sentence_safe_excerpt(chunk.text))
                with preview_col:
                    st.markdown(f"**Page Preview (Page {_display_page_text(chunk)})**")
                    try:
                        st.image(preview_image, use_container_width=True)
                    except Exception:
                        # Fallback for older Streamlit image APIs.
                        st.image(preview_image)
            else:
                st.markdown(_sentence_safe_excerpt(chunk.text))
                st.caption(_preview_unavailable_message(file_type, preview_kind))


def _is_valid_preview_image(preview_image) -> bool:
    """Return True when preview payload looks like a renderable image."""
    if preview_image is None:
        return False

    if isinstance(preview_image, (bytes, bytearray)):
        return len(preview_image) > 0

    if isinstance(preview_image, io.BytesIO):
        return preview_image.getbuffer().nbytes > 0

    if isinstance(preview_image, str):
        return preview_image.strip() not in {"", "0"}

    return False


def _preview_unavailable_message(file_type: str, preview_kind: str) -> str:
    """Return a short user-facing reason when preview cannot be shown."""
    if preview_kind == "pdf_page" or file_type == "pdf":
        return "Preview unavailable for this page."
    return "Preview unavailable for this result type."


def _render_formatted_results_table(query: str, unique) -> None:
    """Render a cleaner table view for quick reading and copying."""
    rows = []
    for idx, (chunk, score) in enumerate(unique, 1):
        rows.append(
            {
                "Rank": idx,
                "Source": chunk.doc_name,
                "Page": _display_page_text(chunk),
                "Relevance": f"{score:.0%}",
                "Snippet": _sentence_safe_excerpt(chunk.text),
            }
        )

    if not rows:
        return

    st.markdown("### Search Results")
    st.caption("Clean, copy-friendly view of where your answer appears.")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _build_export_rows(query: str, unique) -> list:
    """Build serializable row data for CSV/JSON exports."""
    exported_at = datetime.now().strftime("%m/%d/%Y %H:%M")
    rows = []
    for idx, (chunk, score) in enumerate(unique, 1):
        excerpt = _sentence_safe_excerpt(chunk.text)
        rows.append(
            {
                "example query": query,
                "rank": idx,
                "document": chunk.doc_name,
                "page": _display_page_text(chunk),
                "score": round(score, 4),
                "text": excerpt,
                "exported_at": exported_at,
            }
        )
    return rows


def _render_export_actions(export_rows: list) -> None:
    """Render download buttons for CSV and JSON results."""
    st.divider()
    st.markdown("**Export results**")

    export_df = pd.DataFrame(export_rows)
    col1, col2, _ = st.columns([1, 1, 4])
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    with col1:
        st.download_button(
            "Download CSV",
            data=export_df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"search_results_{ts}.csv",
            mime="text/csv",
            type="primary",
            use_container_width=False,
        )
    with col2:
        st.download_button(
            "Download JSON",
            data=json.dumps(export_rows, indent=2, ensure_ascii=False).encode("utf-8"),
            file_name=f"search_results_{ts}.json",
            mime="application/json",
            type="primary",
            use_container_width=False,
        )
