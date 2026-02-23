import streamlit as st
import pandas as pd

from src.config import load_config
from src.email.sender import EmailSender
from gui.components.table_helpers import (
    truncate_text,
    get_primary_category,
)

_TYPE_LABELS = {
    "daily": "Daily",
    "3day": "3-Day",
    "1week": "1-Week",
}


def _build_report_label(entry: dict) -> str:
    """Build a human-readable label for the report selector."""
    run_date = entry.get("run_date", "unknown")
    report_type = entry.get("report_type") or "daily"
    type_label = _TYPE_LABELS.get(report_type, report_type)
    paper_count = entry.get("paper_count", 0) or 0
    matched_count = entry.get("matched_count", 0) or 0
    return f"{run_date} ({type_label}) - {paper_count} papers, {matched_count} matches"


def _split_specific_report(specific: str) -> tuple[str, str]:
    """Split the specific report markdown into synthesis and paper details.

    The report generator joins them with a '---' divider followed by
    '## Paper Details'. Returns (synthesis, paper_details).
    """
    divider = "\n---\n"
    idx = specific.find(divider)
    if idx == -1:
        return specific, ""
    synthesis = specific[:idx]
    paper_details = specific[idx + len(divider) :]
    return synthesis, paper_details


def _render_matches_table(store, matches: list[dict], key_suffix: str = "") -> None:
    """Render matched papers as a compact table with row selection for details."""
    df = pd.DataFrame(
        [
            {
                "#": i,
                "Title": m.get("title", "Unknown"),
                "Score": f"{m.get('llm_score', 0)}/10",
                "Category": get_primary_category(m.get("categories", [])),
                "Relevance": truncate_text(m.get("llm_reason", "N/A"), 80),
                "arXiv": f"https://arxiv.org/abs/{m.get('arxiv_id', '')}",
            }
            for i, m in enumerate(matches, 1)
        ]
    )

    event = st.dataframe(
        df,
        column_config={
            "arXiv": st.column_config.LinkColumn("arXiv", display_text="Open"),
        },
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="multi-row",
        key=f"matches_table{key_suffix}",
    )

    selected_rows = event.selection.rows
    if not selected_rows:
        return

    selected_matches = [matches[i] for i in selected_rows]

    # Export selected matches as CSV
    export_df = _build_match_export_df(selected_matches)
    st.download_button(
        f"Export {len(selected_matches)} paper(s) as CSV",
        export_df.to_csv(index=False),
        file_name="matched_papers.csv",
        mime="text/csv",
        key=f"match_export{key_suffix}",
    )

    # Show details for each selected paper
    for m in selected_matches:
        with st.expander(f"**{m.get('title', 'Unknown')}**", expanded=True):
            _render_match_detail(store, m)


def _build_match_export_df(matches: list[dict]) -> pd.DataFrame:
    """Build a DataFrame for CSV export of matched papers."""
    rows = []
    for m in matches:
        authors = m.get("authors", [])
        if isinstance(authors, list):
            authors = ", ".join(authors)
        categories = m.get("categories", [])
        if isinstance(categories, list):
            categories = ", ".join(categories)
        rows.append(
            {
                "Title": m.get("title", ""),
                "Authors": authors,
                "Categories": categories,
                "LLM Score": m.get("llm_score", ""),
                "Embedding Score": m.get("embedding_score", ""),
                "Relevance": m.get("llm_reason", ""),
                "Abstract": m.get("abstract", ""),
                "arXiv URL": f"https://arxiv.org/abs/{m.get('arxiv_id', '')}",
            }
        )
    return pd.DataFrame(rows)


def _render_match_detail(store, m: dict) -> None:
    """Show full details for a selected matched paper."""
    st.subheader(m.get("title", "Unknown"))

    llm_score = m.get("llm_score", "N/A")
    embedding_score = m.get("embedding_score", 0)
    categories = m.get("categories", [])
    if isinstance(categories, list):
        categories_str = ", ".join(categories)
    else:
        categories_str = str(categories)
    st.markdown(
        f"**Score**: {llm_score}/10 | "
        f"**Embedding**: {embedding_score:.3f} | "
        f"**Categories**: {categories_str}"
    )

    authors = m.get("authors", [])
    if isinstance(authors, list):
        authors_str = ", ".join(authors)
    else:
        authors_str = str(authors)
    st.markdown(f"**Authors**: {authors_str}")

    abstract = m.get("abstract", "")
    if abstract:
        st.markdown(f"**Abstract**: {abstract}")

    reason = m.get("llm_reason", "N/A")
    st.markdown(f"**Why this paper is relevant**: {reason}")

    arxiv_id = m.get("arxiv_id", "")
    if arxiv_id:
        st.markdown(f"[Read on arXiv →](https://arxiv.org/abs/{arxiv_id})")

    # Summary buttons
    paper_id = m.get("paper_id")
    if paper_id:
        st.markdown("**Summaries:**")
        col1, col2, col3, col4 = st.columns(4)
        if col1.button("Brief Summary", key=f"rpt_brief_{paper_id}"):
            _show_summary(store, paper_id, "brief")
        if col2.button("Detailed Summary", key=f"rpt_detailed_{paper_id}"):
            _show_summary(store, paper_id, "detailed")
        if col3.button("中文简要总结", key=f"rpt_brief_zh_{paper_id}"):
            _show_summary(store, paper_id, "brief_zh")
        if col4.button("中文详细总结", key=f"rpt_detailed_zh_{paper_id}"):
            _show_summary(store, paper_id, "detailed_zh")


def _show_summary(store, paper_id: int, mode: str) -> None:
    """Check cache or generate summary for a matched paper."""
    mode_labels = {
        "brief": "brief",
        "detailed": "detailed",
        "brief_zh": "中文简要",
        "detailed_zh": "中文详细",
    }
    label = mode_labels.get(mode, mode)

    cached = store.get_summary(paper_id, mode)
    if cached:
        st.markdown(cached["content"])
    else:
        with st.spinner(f"Generating {label} summary..."):
            import asyncio
            from src.config import load_config
            from src.llm import create_llm_provider
            from src.summarizer.paper_summarizer import PaperSummarizer

            config = load_config()
            llm = create_llm_provider(config)
            summarizer = PaperSummarizer(llm, store)
            try:
                summary = asyncio.run(summarizer.summarize(paper_id, mode))
            except Exception as e:
                st.error(f"Failed to generate summary: {e}")
                st.caption(
                    "If using claude_code provider, verify Claude CLI login via 'claude --print \"hello\"'."
                )
                return
            st.markdown(summary)


def render(store):
    st.title("Reports")

    report_entries = store.get_all_report_entries()
    if not report_entries:
        st.info("No reports generated yet. Run the pipeline first.")
        return

    # Build labels and map index → report id
    labels = [_build_report_label(e) for e in report_entries]
    selected_label = st.selectbox("Select report", labels)
    selected_idx = labels.index(selected_label)
    selected_entry = report_entries[selected_idx]

    report = store.get_report_by_id(selected_entry["id"])
    if not report:
        st.warning("Report not found.")
        return

    run_date = report.get("run_date", "")
    report_type = report.get("report_type") or "daily"
    type_label = _TYPE_LABELS.get(report_type, report_type)

    # Send report via email button
    general = report.get("general_report", "")
    specific = report.get("specific_report", "")
    general_zh = report.get("general_report_zh", "")
    specific_zh = report.get("specific_report_zh", "")

    if general or specific:
        if st.button("Send Report via Email", type="primary"):
            try:
                config = load_config()
                if not config.get("email", {}).get("enabled", False):
                    st.error("Email is not enabled in configuration.")
                else:
                    subject_override = None
                    if report_type != "daily":
                        subject_override = (
                            f"{config['email'].get('subject_prefix', '[Daily Papers]')} "
                            f"{type_label} Report - {run_date}"
                        )
                    sender = EmailSender(config)
                    sender.send_sync(
                        general,
                        specific,
                        run_date,
                        general_zh=general_zh or None,
                        specific_zh=specific_zh or None,
                        subject_override=subject_override,
                    )
                    to_addrs = ", ".join(config["email"]["to"])
                    st.success(f"Report sent to {to_addrs}")
            except Exception as e:
                st.error(f"Failed to send email: {e}")

    # Build tab list: always English, plus Chinese if available
    tab_names = ["General Report", "Specific Report"]
    if general_zh or specific_zh:
        tab_names.extend(["综合报告 (Chinese)", "个性化推荐 (Chinese)"])

    tabs = st.tabs(tab_names)

    with tabs[0]:
        if general:
            st.markdown(general)
        else:
            st.info("No general report for this date.")
    with tabs[1]:
        if specific:
            synthesis, _ = _split_specific_report(specific)

            # Block 1: Theme-based synthesis narrative
            st.markdown(synthesis)

            # Block 2: Matched papers table
            st.divider()
            st.subheader("Paper Details")
            matches = store.get_matches_by_date(run_date)
            if matches:
                _render_matches_table(store, matches, key_suffix="_en")
            else:
                st.info("No matched papers for this report.")
        else:
            st.info("No specific report for this date.")

    # Chinese tabs (if available)
    if general_zh or specific_zh:
        with tabs[2]:
            if general_zh:
                st.markdown(general_zh)
            else:
                st.info("本日暂无中文综合报告。")
        with tabs[3]:
            if specific_zh:
                synthesis_zh, _ = _split_specific_report(specific_zh)
                st.markdown(synthesis_zh)

                st.divider()
                st.subheader("论文详情")
                matches = store.get_matches_by_date(run_date)
                if matches:
                    _render_matches_table(store, matches, key_suffix="_zh")
                else:
                    st.info("暂无匹配论文。")
            else:
                st.info("本日暂无中文个性化推荐报告。")
