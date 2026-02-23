import streamlit as st
from datetime import date, timedelta


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

    # Multi-day report buttons
    st.divider()
    st.subheader("Multi-Day Reports")

    col_3day, col_1week = st.columns(2)

    with col_3day:
        if st.button("Generate 3-Day Report", use_container_width=True):
            _run_range_report(days=3, report_type="3day")

    with col_1week:
        if st.button("Generate 1-Week Report", use_container_width=True):
            _run_range_report(days=7, report_type="1week")


def _run_range_report(days: int, report_type: str):
    """Run range report pipeline with spinner and feedback."""
    label = "3-day" if report_type == "3day" else "1-week"
    with st.spinner(f"Generating {label} report... This may take several minutes."):
        import asyncio
        from src.pipeline import DailyPipeline
        from src.config import load_config

        end = date.today()
        start = end - timedelta(days=days - 1)
        config = load_config()
        pipeline = DailyPipeline(config)
        result = asyncio.run(
            pipeline.run_range_report(start.isoformat(), end.isoformat(), report_type)
        )
        if result["papers_count"] == 0:
            st.warning(f"No papers found in the last {days} days.")
        else:
            st.success(
                f"{label.capitalize()} report generated: "
                f"{result['papers_count']} papers, {result['matches']} matches"
            )
        st.rerun()
