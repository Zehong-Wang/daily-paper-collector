import streamlit as st


def render_report(report_markdown: str):
    """Render a Markdown report in Streamlit."""
    st.markdown(report_markdown)
