import streamlit as st
import pandas as pd
import numpy as np
from datetime import date

from gui.components.table_helpers import (
    truncate_authors,
    get_primary_category,
)


def _compute_relevance_scores(papers, store):
    """Compute BERT embedding similarity scores for papers against user interests.

    Returns a dict mapping arxiv_id -> max cosine similarity score.
    """
    interests = store.get_interests_with_embeddings()
    if not interests:
        return {}

    interest_vecs = [
        np.frombuffer(i["embedding"], dtype=np.float32)
        for i in interests
        if i.get("embedding")
    ]
    if not interest_vecs:
        return {}
    interest_matrix = np.array(interest_vecs)

    paper_vecs = []
    paper_ids = []
    for p in papers:
        if p.get("embedding"):
            paper_vecs.append(np.frombuffer(p["embedding"], dtype=np.float32))
            paper_ids.append(p.get("arxiv_id", ""))

    if not paper_vecs:
        return {}

    paper_matrix = np.array(paper_vecs)
    # Embeddings are already normalized, so dot product = cosine similarity
    similarity = paper_matrix @ interest_matrix.T
    max_scores = similarity.max(axis=1)

    return {aid: float(score) for aid, score in zip(paper_ids, max_scores)}


def render(store):
    st.title("Papers")

    # Date selector
    selected_date = st.date_input("Select date", value=date.today())
    date_str = selected_date.isoformat()

    # Search box
    search_query = st.text_input("Search papers (title or abstract)")

    if search_query:
        papers = store.search_papers(search_query)
        st.info(f"Found {len(papers)} results for '{search_query}'")
    else:
        papers = store.get_papers_by_date(date_str)
        st.info(f"{len(papers)} papers on {date_str}")

    if not papers:
        return

    # Compute BERT embedding relevance scores against user interests
    relevance_map = _compute_relevance_scores(papers, store)

    # Build table data
    df = pd.DataFrame(
        [
            {
                "Title": p["title"],
                "Authors": truncate_authors(p["authors"]),
                "Category": get_primary_category(p["categories"]),
                "Relevance": f"{relevance_map[aid]:.3f}" if (aid := p.get("arxiv_id", "")) in relevance_map else "-",
                "Date": p["published_date"],
                "arXiv": p.get("pdf_url", ""),
            }
            for p in papers
        ]
    )

    # Render table with multi-row selection
    event = st.dataframe(
        df,
        column_config={
            "arXiv": st.column_config.LinkColumn("arXiv", display_text="Open"),
        },
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="multi-row",
    )

    selected_rows = event.selection.rows
    if not selected_rows:
        return

    selected_papers = [papers[i] for i in selected_rows]

    # Export selected papers as CSV
    export_df = _build_export_df(selected_papers)
    st.download_button(
        f"Export {len(selected_papers)} paper(s) as CSV",
        export_df.to_csv(index=False),
        file_name=f"papers_{date_str}.csv",
        mime="text/csv",
    )

    # Show details for each selected paper
    for paper in selected_papers:
        with st.expander(f"**{paper['title']}**", expanded=True):
            _render_detail_panel(store, paper)


def _build_export_df(papers: list[dict]) -> pd.DataFrame:
    """Build a DataFrame for CSV export."""
    rows = []
    for p in papers:
        authors = p["authors"]
        if isinstance(authors, list):
            authors = ", ".join(authors)
        categories = p["categories"]
        if isinstance(categories, list):
            categories = ", ".join(categories)
        rows.append(
            {
                "Title": p["title"],
                "Authors": authors,
                "Categories": categories,
                "Published": p["published_date"],
                "Abstract": p.get("abstract", ""),
                "arXiv URL": p.get("pdf_url", ""),
            }
        )
    return pd.DataFrame(rows)


def _render_detail_panel(store, paper):
    """Show full paper details and summary buttons below the table."""
    st.subheader(paper["title"])

    authors = paper["authors"]
    if isinstance(authors, list):
        authors = ", ".join(authors)
    st.markdown(f"**Authors:** {authors}")

    categories = paper["categories"]
    if isinstance(categories, list):
        categories = ", ".join(categories)
    st.markdown(f"**Categories:** {categories}")

    st.markdown(f"**Published:** {paper['published_date']}")
    st.markdown(paper["abstract"])
    st.markdown(f"[Read on arXiv →]({paper.get('pdf_url', '#')})")

    # Summarize buttons
    col1, col2 = st.columns(2)
    if col1.button("Brief Summary", key=f"brief_{paper['id']}"):
        _show_summary(store, paper, "brief")
    if col2.button("Detailed Summary", key=f"detailed_{paper['id']}"):
        _show_summary(store, paper, "detailed")


def _show_summary(store, paper, mode):
    """Check cache or generate summary."""
    cached = store.get_summary(paper["id"], mode)
    if cached:
        st.markdown(cached["content"])
    else:
        with st.spinner(f"Generating {mode} summary..."):
            import asyncio
            from src.config import load_config
            from src.llm import create_llm_provider
            from src.summarizer.paper_summarizer import PaperSummarizer

            config = load_config()
            llm = create_llm_provider(config)
            summarizer = PaperSummarizer(llm, store)
            try:
                summary = asyncio.run(summarizer.summarize(paper["id"], mode))
            except Exception as e:
                st.error(f"Failed to generate summary: {e}")
                st.caption(
                    "If using claude_code provider, verify Claude CLI login via 'claude --print \"hello\"'."
                )
                return
            st.markdown(summary)
