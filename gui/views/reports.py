import streamlit as st

from src.config import load_config
from src.email.sender import EmailSender


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


def _render_paper_cards(matches: list[dict]) -> None:
    """Render expandable paper cards with comprehensive details."""
    for i, m in enumerate(matches, 1):
        title = m.get("title", "Unknown")
        llm_score = m.get("llm_score", "N/A")
        embedding_score = m.get("embedding_score", 0)
        label = f"**{i}. {title}** — Score: {llm_score}/10"
        with st.expander(label):
            # Score and categories
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

            # Authors (full, no truncation)
            authors = m.get("authors", [])
            if isinstance(authors, list):
                authors_str = ", ".join(authors)
            else:
                authors_str = str(authors)
            st.markdown(f"**Authors**: {authors_str}")

            # Abstract (full, no truncation)
            abstract = m.get("abstract", "")
            if abstract:
                st.markdown(f"**Abstract**: {abstract}")

            # Relevance reason
            reason = m.get("llm_reason", "N/A")
            st.markdown(f"**Why this paper is relevant**: {reason}")

            # arXiv link
            arxiv_id = m.get("arxiv_id", "")
            if arxiv_id:
                st.markdown(f"[Read on arXiv →](https://arxiv.org/abs/{arxiv_id})")


def render(store):
    st.title("Reports")

    report_dates = store.get_all_report_dates()
    if not report_dates:
        st.info("No reports generated yet. Run the pipeline first.")
        return

    selected_date = st.selectbox("Select report date", report_dates)
    report = store.get_report_by_date(selected_date)

    if report:
        # Send report via email button
        general = report.get("general_report", "")
        specific = report.get("specific_report", "")

        if general or specific:
            if st.button("Send Report via Email", type="primary"):
                try:
                    config = load_config()
                    if not config.get("email", {}).get("enabled", False):
                        st.error("Email is not enabled in configuration.")
                    else:
                        sender = EmailSender(config)
                        sender.send_sync(general, specific, selected_date)
                        to_addrs = ", ".join(config["email"]["to"])
                        st.success(f"Report sent to {to_addrs}")
                except Exception as e:
                    st.error(f"Failed to send email: {e}")

        tab1, tab2 = st.tabs(["General Report", "Specific Report"])
        with tab1:
            if general:
                st.markdown(general)
            else:
                st.info("No general report for this date.")
        with tab2:
            if specific:
                synthesis, _ = _split_specific_report(specific)

                # Block 1: Theme-based synthesis narrative
                st.markdown(synthesis)

                # Block 2: Expandable paper-wise cards
                st.divider()
                st.subheader("Paper Details")
                matches = store.get_matches_by_date(selected_date)
                if matches:
                    _render_paper_cards(matches)
                else:
                    st.info("No matched papers for this date.")
            else:
                st.info("No specific report for this date.")
