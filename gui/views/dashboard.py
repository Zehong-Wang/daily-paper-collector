import streamlit as st
from datetime import date


def render(store):
    st.title("Dashboard")

    today = date.today().isoformat()

    # Row 1: Metrics
    col1, col2, col3 = st.columns(3)
    papers_today = store.get_papers_by_date(today)
    report = store.get_report_by_date(today)
    matches = store.get_matches_by_date(today)

    col1.metric("Papers Today", len(papers_today))
    col2.metric("Matches Today", len(matches))
    col3.metric("Reports", len(store.get_all_report_dates()))

    # Row 2: Latest General Report preview
    if report and report.get("general_report"):
        st.subheader("Latest General Report")
        preview = report["general_report"]
        if len(preview) > 1000:
            preview = preview[:1000] + "..."
        st.markdown(preview)

    # Row 3: Latest Specific Report preview
    if report and report.get("specific_report"):
        st.subheader("Latest Specific Report")
        preview = report["specific_report"]
        if len(preview) > 1000:
            preview = preview[:1000] + "..."
        st.markdown(preview)

    # Manual trigger button
    if st.button("Run Pipeline Now"):
        with st.spinner("Running pipeline..."):
            import asyncio
            from src.pipeline import DailyPipeline
            from src.config import load_config

            config = load_config()
            pipeline = DailyPipeline(config)
            result = asyncio.run(pipeline.run())
            st.success(f"Pipeline completed: {result}")
            st.rerun()
