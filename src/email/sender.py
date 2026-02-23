import asyncio
import logging
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import markdown
import premailer


class EmailSender:
    def __init__(self, config: dict):
        self.logger = logging.getLogger(__name__)
        email_config = config["email"]
        smtp_config = email_config["smtp"]

        self.host = smtp_config["host"]
        self.port = smtp_config["port"]
        self.username = os.environ.get(smtp_config["username_env"], "")
        self.password = os.environ.get(smtp_config["password_env"], "")
        self.from_address = email_config["from"]
        self.to_addresses = email_config["to"]
        self.subject_prefix = email_config.get("subject_prefix", "[Daily Papers]")

    def render_markdown_to_html(self, md_content: str) -> str:
        """Convert Markdown to HTML and inline CSS.

        1. Convert Markdown to HTML with tables and fenced_code extensions.
        2. Wrap in a basic HTML template with a <style> block for readability.
        3. Inline CSS via premailer.transform().

        Returns the final HTML string with inlined styles.
        """
        html_body = markdown.markdown(md_content, extensions=["tables", "fenced_code"])

        html_template = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    line-height: 1.7;
    color: #2d3748;
    background-color: #f7fafc;
    margin: 0;
    padding: 0;
}}
.wrapper {{
    max-width: 720px;
    margin: 0 auto;
    padding: 24px 16px;
}}
h1 {{
    font-size: 26px;
    font-weight: 700;
    color: #1a202c;
    border-bottom: 3px solid #4299e1;
    padding-bottom: 12px;
    margin-bottom: 24px;
}}
h2 {{
    font-size: 20px;
    font-weight: 600;
    color: #2b6cb0;
    margin-top: 32px;
    margin-bottom: 16px;
    padding: 10px 14px;
    background-color: #ebf8ff;
    border-left: 4px solid #4299e1;
    border-radius: 0 4px 4px 0;
}}
h3 {{
    font-size: 17px;
    font-weight: 600;
    color: #2d3748;
    margin-top: 24px;
    margin-bottom: 10px;
}}
p {{
    margin: 8px 0 12px 0;
}}
a {{
    color: #2b6cb0;
    text-decoration: none;
    font-weight: 500;
}}
ul, ol {{
    padding-left: 20px;
    margin: 8px 0 16px 0;
}}
li {{
    margin-bottom: 10px;
    line-height: 1.6;
}}
table {{
    border-collapse: collapse;
    width: auto;
    margin: 12px 0 20px 0;
    font-size: 14px;
}}
th {{
    background-color: #edf2f7;
    color: #2d3748;
    font-weight: 600;
    padding: 8px 16px;
    border: 1px solid #e2e8f0;
    text-align: left;
}}
td {{
    padding: 6px 16px;
    border: 1px solid #e2e8f0;
}}
tr:nth-child(even) td {{
    background-color: #f7fafc;
}}
hr {{
    border: none;
    border-top: 2px solid #e2e8f0;
    margin: 32px 0;
}}
blockquote {{
    margin: 4px 0 12px 0;
    padding: 8px 14px;
    background-color: #f0fff4;
    border-left: 3px solid #48bb78;
    color: #276749;
    font-size: 14px;
    border-radius: 0 4px 4px 0;
}}
blockquote p {{
    margin: 0;
}}
code {{
    background-color: #edf2f7;
    padding: 2px 6px;
    border-radius: 3px;
    font-size: 13px;
    color: #553c9a;
}}
strong {{
    color: #1a202c;
}}
</style>
</head>
<body>
<div class="wrapper">
{html_body}
</div>
</body>
</html>"""

        return premailer.transform(html_template)

    def _combine_reports(
        self,
        general_report: str,
        specific_report: str,
        general_zh: str = None,
        specific_zh: str = None,
    ) -> str:
        """Combine English and optional Chinese reports into a single Markdown string."""
        parts = [general_report, "---", specific_report]

        if general_zh or specific_zh:
            parts.append("---")
            if general_zh:
                parts.append(general_zh)
            if specific_zh:
                parts.append("---")
                parts.append(specific_zh)

        return "\n\n".join(parts)

    def send_sync(
        self,
        general_report: str,
        specific_report: str,
        run_date: str,
        general_zh: str = None,
        specific_zh: str = None,
    ):
        """Synchronous version of send() for non-async contexts (e.g. Streamlit GUI)."""
        self.logger.info("Preparing email for %s", run_date)
        combined_md = self._combine_reports(
            general_report, specific_report, general_zh, specific_zh
        )
        html_content = self.render_markdown_to_html(combined_md)
        subject = f"{self.subject_prefix} {run_date}"
        msg = self._build_email(html_content, subject)
        self._send_smtp(msg)
        self.logger.info("Email sent successfully to %s", ", ".join(self.to_addresses))

    async def send(
        self,
        general_report: str,
        specific_report: str,
        ranked_papers: list[dict],
        run_date: str,
        general_zh: str = None,
        specific_zh: str = None,
    ):
        """Assemble and send the daily email.

        1. Combine English + Chinese reports into one Markdown string.
        2. Convert to HTML via render_markdown_to_html().
        3. Create MIMEMultipart message with subject: "{subject_prefix} {run_date}".
        4. Attach HTML as MIMEText("...", "html").
        5. Connect to SMTP, starttls, login, send.

        Uses asyncio.to_thread() to wrap the synchronous smtplib calls.
        """
        self.logger.info("Preparing email for %s", run_date)

        # Combine reports
        combined_md = self._combine_reports(
            general_report, specific_report, general_zh, specific_zh
        )

        # Convert to HTML
        html_content = self.render_markdown_to_html(combined_md)

        # Build email
        subject = f"{self.subject_prefix} {run_date}"
        msg = self._build_email(html_content, subject)

        # Send via SMTP in a thread to avoid blocking the event loop
        await asyncio.to_thread(self._send_smtp, msg)

        self.logger.info("Email sent successfully to %s", ", ".join(self.to_addresses))

    def _build_email(self, html_content: str, subject: str) -> MIMEMultipart:
        """Build the MIME message object."""
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self.from_address
        msg["To"] = ", ".join(self.to_addresses)

        html_part = MIMEText(html_content, "html")
        msg.attach(html_part)

        return msg

    def _send_smtp(self, msg: MIMEMultipart):
        """Send the email via SMTP. Called from a thread.

        Catches smtplib.SMTPException, logs the error, and re-raises
        so the caller (pipeline) can handle the failure.
        """
        try:
            with smtplib.SMTP(self.host, self.port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
        except smtplib.SMTPException as e:
            self.logger.error(f"SMTP error: {e}")
            raise
