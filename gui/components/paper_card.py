import streamlit as st


def render_paper_card(paper: dict):
    """Render a single paper card inside an expander."""
    with st.expander(f"**{paper['title']}**"):
        authors = paper["authors"]
        if isinstance(authors, list):
            authors = ", ".join(authors)
        st.write(f"**Authors:** {authors}")

        categories = paper["categories"]
        if isinstance(categories, list):
            categories = ", ".join(categories)
        st.write(f"**Categories:** {categories}")

        abstract = paper.get("abstract", "")
        if len(abstract) > 300:
            abstract = abstract[:300] + "..."
        st.markdown(abstract)
        st.markdown(f"[arXiv]({paper.get('pdf_url', '#')})")
