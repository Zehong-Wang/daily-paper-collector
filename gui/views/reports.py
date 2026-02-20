import streamlit as st

from src.config import load_config
from src.email.sender import EmailSender


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
                st.markdown(specific)
            else:
                st.info("No specific report for this date.")

        # Show matches for this date
        st.subheader("Match Results")
        matches = store.get_matches_by_date(selected_date)
        if matches:
            for m in matches:
                llm_score = m.get("llm_score", "N/A")
                embedding_score = m.get("embedding_score", 0)
                with st.expander(
                    f"**{m['title']}** (LLM: {llm_score}/10, Embedding: {embedding_score:.3f})"
                ):
                    st.write(f"**Reason:** {m.get('llm_reason', 'N/A')}")
                    abstract = m.get("abstract", "")
                    if len(abstract) > 300:
                        abstract = abstract[:300] + "..."
                    st.markdown(abstract)
        else:
            st.info("No matches for this date.")
