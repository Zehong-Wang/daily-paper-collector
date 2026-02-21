"""Cached Streamlit resources shared across GUI pages.

Separated from app.py to allow safe imports without re-triggering
the navigation runner (which would cause nested form errors).
"""

import streamlit as st

from src.config import load_config
from src.store.database import PaperStore
from src.matcher.embedder import Embedder


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
