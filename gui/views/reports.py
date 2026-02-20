import streamlit as st


def render(store):
    st.title("Reports")

    report_dates = store.get_all_report_dates()
    if not report_dates:
        st.info("No reports generated yet. Run the pipeline first.")
        return

    selected_date = st.selectbox("Select report date", report_dates)
    report = store.get_report_by_date(selected_date)

    if report:
        tab1, tab2 = st.tabs(["General Report", "Specific Report"])
        with tab1:
            if report.get("general_report"):
                st.markdown(report["general_report"])
            else:
                st.info("No general report for this date.")
        with tab2:
            if report.get("specific_report"):
                st.markdown(report["specific_report"])
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
