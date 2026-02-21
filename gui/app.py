import streamlit as st
import sys
import os

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import setup_logging
from gui.state import get_config, get_store, get_embedder  # noqa: F401 — re-exported for backwards compat

setup_logging()

st.set_page_config(page_title="Daily Paper Collector", layout="wide")


def _make_page(module_path: str):
    """Create a page callable that imports and renders the given view module."""
    module_name = module_path.rsplit(".", 1)[-1]

    def page_fn():
        import importlib
        mod = importlib.import_module(module_path)
        mod.render(get_store())

    # Streamlit may infer pathnames from callable names; keep names unique per page.
    page_fn.__name__ = f"render_{module_name}"
    page_fn.__qualname__ = page_fn.__name__
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
