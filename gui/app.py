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


st.set_page_config(page_title="Daily Paper Collector", layout="wide")


def _make_page(module_path: str):
    """Create a page callable that imports and renders the given view module."""
    def page_fn():
        import importlib
        mod = importlib.import_module(module_path)
        mod.render(get_store())
    return page_fn


pages = [
    st.Page(_make_page("gui.views.dashboard"), title="Dashboard", url_path="dashboard", default=True),
    st.Page(_make_page("gui.views.papers"), title="Papers", url_path="papers"),
    st.Page(_make_page("gui.views.interests"), title="Interests", url_path="interests"),
    st.Page(_make_page("gui.views.reports"), title="Reports", url_path="reports"),
    st.Page(_make_page("gui.views.settings"), title="Settings", url_path="settings"),
]

nav = st.navigation(pages)
nav.run()
