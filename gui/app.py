import streamlit as st
import sys
import os

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config, setup_logging
from src.store.database import PaperStore
from src.matcher.embedder import Embedder

setup_logging()


@st.cache_resource
def get_config() -> dict:
    """Cached config dict. Cleared on app restart."""
    return load_config()


@st.cache_resource
def get_store() -> PaperStore:
    """Cached PaperStore instance. Avoids recreating DB connections on every rerun."""
    config = get_config()
    return PaperStore(config["database"]["path"])


@st.cache_resource
def get_embedder() -> Embedder:
    """Cached Embedder instance. Avoids reloading the ~80MB sentence-transformer model
    on every Streamlit rerun."""
    config = get_config()
    return Embedder(config)


def main():
    st.set_page_config(page_title="Daily Paper Collector", layout="wide")

    page = st.sidebar.radio(
        "Navigation", ["Dashboard", "Papers", "Interests", "Reports", "Settings"]
    )

    if page == "Dashboard":
        from gui.pages.dashboard import render
    elif page == "Papers":
        from gui.pages.papers import render
    elif page == "Interests":
        from gui.pages.interests import render
    elif page == "Reports":
        from gui.pages.reports import render
    elif page == "Settings":
        from gui.pages.settings import render

    render(get_store())


if __name__ == "__main__":
    main()
