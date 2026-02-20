import streamlit as st


def render(store):
    st.title("Interest Management")

    # Show existing interests
    interests = store.get_all_interests()
    if interests:
        st.subheader(f"Current Interests ({len(interests)})")
        for interest in interests:
            col1, col2, col3 = st.columns([3, 1, 1])
            label = f"**[{interest['type']}]** {interest['value']}"
            if interest.get("description"):
                label += f" — {interest['description']}"
            col1.write(label)
            has_emb = "Y" if interest.get("embedding") else "N"
            col2.write(f"Embedding: {has_emb}")
            if col3.button("Delete", key=f"del_{interest['id']}"):
                store.delete_interest(interest["id"])
                st.rerun()
    else:
        st.info("No interests configured yet. Add some below.")

    # Add new interest form
    st.subheader("Add New Interest")
    with st.form("add_interest"):
        interest_type = st.selectbox("Type", ["keyword", "paper", "reference_paper"])
        value = st.text_input("Value (keyword text or arXiv ID)")
        description = st.text_input("Description (optional)")
        submitted = st.form_submit_button("Add Interest")

        if submitted and value:
            from gui.app import get_embedder
            from src.interest.manager import InterestManager

            embedder = get_embedder()
            mgr = InterestManager(store, embedder)

            if interest_type == "keyword":
                mgr.add_keyword(value, description or None)
            elif interest_type == "paper":
                mgr.add_paper(value, description or None)
            else:
                mgr.add_reference_paper(value, description or None)
            st.success(f"Added {interest_type}: {value}")
            st.rerun()
