import streamlit as st
from datetime import date


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

    # Display papers
    for paper in papers:
        with st.expander(f"**{paper['title']}**"):
            authors = paper["authors"]
            if isinstance(authors, list):
                authors = ", ".join(authors)
            st.write(f"**Authors:** {authors}")

            categories = paper["categories"]
            if isinstance(categories, list):
                categories = ", ".join(categories)
            st.write(f"**Categories:** {categories}")

            st.write(f"**Published:** {paper['published_date']}")
            st.markdown(paper["abstract"])
            st.markdown(f"[arXiv]({paper['pdf_url']})")

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
            summary = asyncio.run(summarizer.summarize(paper["id"], mode))
            st.markdown(summary)
