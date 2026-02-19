"""Tests for EmailSender (Phase 8)."""

import asyncio
from email.mime.multipart import MIMEMultipart
from unittest.mock import MagicMock, patch

import pytest

from src.email.sender import EmailSender


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_config(
    host="smtp.gmail.com",
    port=587,
    from_addr="sender@test.com",
    to_addrs=None,
    subject_prefix="[Daily Papers]",
):
    """Create a minimal email config dict."""
    return {
        "email": {
            "enabled": True,
            "smtp": {
                "host": host,
                "port": port,
                "username_env": "EMAIL_USERNAME",
                "password_env": "EMAIL_PASSWORD",
            },
            "from": from_addr,
            "to": to_addrs or ["recipient@test.com"],
            "subject_prefix": subject_prefix,
        }
    }


@pytest.fixture
def config():
    return _make_config()


@pytest.fixture
def sender(config, monkeypatch):
    monkeypatch.setenv("EMAIL_USERNAME", "testuser")
    monkeypatch.setenv("EMAIL_PASSWORD", "testpass")
    return EmailSender(config)


# ---------------------------------------------------------------------------
# TestInit
# ---------------------------------------------------------------------------


class TestInit:
    def test_reads_smtp_settings(self, sender):
        assert sender.host == "smtp.gmail.com"
        assert sender.port == 587

    def test_reads_credentials_from_env(self, sender):
        assert sender.username == "testuser"
        assert sender.password == "testpass"

    def test_reads_addresses(self, sender):
        assert sender.from_address == "sender@test.com"
        assert sender.to_addresses == ["recipient@test.com"]

    def test_reads_subject_prefix(self, sender):
        assert sender.subject_prefix == "[Daily Papers]"

    def test_custom_subject_prefix(self, monkeypatch):
        monkeypatch.setenv("EMAIL_USERNAME", "u")
        monkeypatch.setenv("EMAIL_PASSWORD", "p")
        config = _make_config(subject_prefix="[Papers]")
        s = EmailSender(config)
        assert s.subject_prefix == "[Papers]"

    def test_multiple_recipients(self, monkeypatch):
        monkeypatch.setenv("EMAIL_USERNAME", "u")
        monkeypatch.setenv("EMAIL_PASSWORD", "p")
        config = _make_config(to_addrs=["a@test.com", "b@test.com"])
        s = EmailSender(config)
        assert s.to_addresses == ["a@test.com", "b@test.com"]


# ---------------------------------------------------------------------------
# TestRenderMarkdownToHtml
# ---------------------------------------------------------------------------


class TestRenderMarkdownToHtml:
    def test_heading_rendered(self, sender):
        result = sender.render_markdown_to_html("# Hello")
        assert "<h1" in result
        assert "Hello" in result

    def test_bold_rendered(self, sender):
        result = sender.render_markdown_to_html("**bold**")
        # premailer inlines CSS onto <strong>, so check without closing bracket
        assert "<strong" in result
        assert "bold" in result

    def test_inline_css_applied(self, sender):
        result = sender.render_markdown_to_html("# Hello\n\n**bold**")
        assert 'style="' in result

    def test_table_rendered(self, sender):
        md = "| A | B |\n|---|---|\n| 1 | 2 |"
        result = sender.render_markdown_to_html(md)
        assert "<table" in result
        assert "<td" in result

    def test_link_rendered(self, sender):
        md = "[arXiv](https://arxiv.org/abs/2501.12345)"
        result = sender.render_markdown_to_html(md)
        assert "https://arxiv.org/abs/2501.12345" in result
        assert "<a" in result

    def test_horizontal_rule_rendered(self, sender):
        md = "above\n\n---\n\nbelow"
        result = sender.render_markdown_to_html(md)
        assert "<hr" in result

    def test_bullet_list_rendered(self, sender):
        md = "- item one\n- item two"
        result = sender.render_markdown_to_html(md)
        assert "<li>" in result or "<ul>" in result


# ---------------------------------------------------------------------------
# TestBuildEmail
# ---------------------------------------------------------------------------


class TestBuildEmail:
    def test_returns_mime_multipart(self, sender):
        msg = sender._build_email("<html><body>Hi</body></html>", "Test Subject")
        assert isinstance(msg, MIMEMultipart)

    def test_subject_is_set(self, sender):
        msg = sender._build_email("<html></html>", "[Daily Papers] 2025-01-15")
        assert msg["Subject"] == "[Daily Papers] 2025-01-15"

    def test_from_address(self, sender):
        msg = sender._build_email("<html></html>", "Test")
        assert msg["From"] == "sender@test.com"

    def test_to_address(self, sender):
        msg = sender._build_email("<html></html>", "Test")
        assert msg["To"] == "recipient@test.com"

    def test_multiple_recipients_in_to(self, monkeypatch):
        monkeypatch.setenv("EMAIL_USERNAME", "u")
        monkeypatch.setenv("EMAIL_PASSWORD", "p")
        config = _make_config(to_addrs=["a@test.com", "b@test.com"])
        s = EmailSender(config)
        msg = s._build_email("<html></html>", "Test")
        assert msg["To"] == "a@test.com, b@test.com"

    def test_html_payload_attached(self, sender):
        html = "<html><body><h1>Hello</h1></body></html>"
        msg = sender._build_email(html, "Test")
        payloads = msg.get_payload()
        assert len(payloads) == 1
        assert payloads[0].get_content_type() == "text/html"
        assert "Hello" in payloads[0].get_payload(decode=True).decode()


# ---------------------------------------------------------------------------
# TestSend
# ---------------------------------------------------------------------------


class TestSend:
    @pytest.mark.asyncio
    async def test_smtp_called_correctly(self, sender):
        """Verify that SMTP starttls, login, and send_message are called."""
        mock_smtp_instance = MagicMock()
        mock_smtp_instance.__enter__ = MagicMock(return_value=mock_smtp_instance)
        mock_smtp_instance.__exit__ = MagicMock(return_value=False)

        with patch("src.email.sender.smtplib.SMTP", return_value=mock_smtp_instance):
            await sender.send(
                "# General\n\nOverview",
                "## Specific\n\nDetails",
                [],
                "2025-01-15",
            )

        mock_smtp_instance.starttls.assert_called_once()
        mock_smtp_instance.login.assert_called_once_with("testuser", "testpass")
        mock_smtp_instance.send_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_message_has_correct_subject(self, sender):
        """Verify the sent message has the expected subject line."""
        mock_smtp_instance = MagicMock()
        mock_smtp_instance.__enter__ = MagicMock(return_value=mock_smtp_instance)
        mock_smtp_instance.__exit__ = MagicMock(return_value=False)

        with patch("src.email.sender.smtplib.SMTP", return_value=mock_smtp_instance):
            await sender.send("# General", "## Specific", [], "2025-01-15")

        sent_msg = mock_smtp_instance.send_message.call_args[0][0]
        assert sent_msg["Subject"] == "[Daily Papers] 2025-01-15"

    @pytest.mark.asyncio
    async def test_send_combines_reports(self, sender):
        """Verify that general and specific reports are both present in the email."""
        mock_smtp_instance = MagicMock()
        mock_smtp_instance.__enter__ = MagicMock(return_value=mock_smtp_instance)
        mock_smtp_instance.__exit__ = MagicMock(return_value=False)

        with patch("src.email.sender.smtplib.SMTP", return_value=mock_smtp_instance):
            await sender.send(
                "# General Report Content",
                "## Specific Report Content",
                [],
                "2025-01-15",
            )

        sent_msg = mock_smtp_instance.send_message.call_args[0][0]
        payload = sent_msg.get_payload()[0].get_payload(decode=True).decode()
        assert "General Report Content" in payload
        assert "Specific Report Content" in payload

    @pytest.mark.asyncio
    async def test_no_actual_email_sent(self, sender):
        """Verify mock prevents real email sending."""
        with patch("src.email.sender.smtplib.SMTP") as mock_smtp_cls:
            mock_instance = MagicMock()
            mock_instance.__enter__ = MagicMock(return_value=mock_instance)
            mock_instance.__exit__ = MagicMock(return_value=False)
            mock_smtp_cls.return_value = mock_instance

            await sender.send("# Report", "## Details", [], "2025-01-15")

            mock_smtp_cls.assert_called_once_with("smtp.gmail.com", 587)

    @pytest.mark.asyncio
    async def test_smtp_host_and_port(self, monkeypatch):
        """Verify custom SMTP host and port are used."""
        monkeypatch.setenv("EMAIL_USERNAME", "u")
        monkeypatch.setenv("EMAIL_PASSWORD", "p")
        config = _make_config(host="smtp.custom.com", port=465)
        s = EmailSender(config)

        mock_instance = MagicMock()
        mock_instance.__enter__ = MagicMock(return_value=mock_instance)
        mock_instance.__exit__ = MagicMock(return_value=False)

        with patch("src.email.sender.smtplib.SMTP", return_value=mock_instance) as mock_cls:
            await s.send("# Report", "## Details", [], "2025-01-15")
            mock_cls.assert_called_once_with("smtp.custom.com", 465)
