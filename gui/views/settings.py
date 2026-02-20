import streamlit as st
import yaml


def render(store):
    st.title("Settings")

    from src.config import load_config, get_project_root

    config = load_config()

    # Display current config (read-only for safety)
    st.subheader("Current Configuration")
    st.code(yaml.dump(config, default_flow_style=False), language="yaml")

    # Editable sections
    st.subheader("ArXiv Categories")
    categories = st.text_area(
        "Categories (one per line)",
        value="\n".join(config.get("arxiv", {}).get("categories", [])),
    )

    st.subheader("LLM Provider")
    providers = ["openai", "claude", "claude_code"]
    current_provider = config.get("llm", {}).get("provider", "openai")
    provider_index = providers.index(current_provider) if current_provider in providers else 0
    provider = st.selectbox("Provider", providers, index=provider_index)

    st.subheader("Email")
    email_enabled = st.checkbox("Enable email", value=config.get("email", {}).get("enabled", False))

    if st.button("Save Settings"):
        config["arxiv"]["categories"] = [c.strip() for c in categories.split("\n") if c.strip()]
        config["llm"]["provider"] = provider
        config["email"]["enabled"] = email_enabled
        config_path = get_project_root() / "config" / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        st.success("Settings saved!")

    # Email test
    st.subheader("Test Email")
    if st.button("Send Test Email"):
        with st.spinner("Sending test email..."):
            try:
                import asyncio
                from src.email.sender import EmailSender

                sender = EmailSender(config)
                asyncio.run(
                    sender.send(
                        "# Test Email\n\nThis is a test.",
                        "## No specific report",
                        [],
                        "test",
                    )
                )
                st.success("Test email sent!")
            except Exception as e:
                st.error(f"Failed: {e}")
