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
<style>
body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    line-height: 1.6;
    color: #333333;
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
}}
h1 {{
    font-size: 24px;
    color: #1a1a1a;
    border-bottom: 2px solid #e0e0e0;
    padding-bottom: 8px;
}}
h2 {{
    font-size: 20px;
    color: #2c2c2c;
    margin-top: 24px;
}}
h3 {{
    font-size: 16px;
    color: #444444;
    margin-top: 16px;
}}
a {{
    color: #0066cc;
    text-decoration: none;
}}
a:hover {{
    text-decoration: underline;
}}
table {{
    border-collapse: collapse;
    width: 100%;
    margin: 16px 0;
}}
th, td {{
    border: 1px solid #dddddd;
    padding: 8px 12px;
    text-align: left;
}}
th {{
    background-color: #f5f5f5;
}}
hr {{
    border: none;
    border-top: 1px solid #e0e0e0;
    margin: 24px 0;
}}
code {{
    background-color: #f5f5f5;
    padding: 2px 6px;
    border-radius: 3px;
    font-size: 14px;
}}
strong {{
    color: #1a1a1a;
}}
</style>
</head>
<body>
{html_body}
</body>
</html>"""

        return premailer.transform(html_template)

    async def send(
        self,
        general_report: str,
        specific_report: str,
        ranked_papers: list[dict],
        run_date: str,
    ):
        """Assemble and send the daily email.

        1. Combine general_report + specific_report into one Markdown string.
        2. Convert to HTML via render_markdown_to_html().
        3. Create MIMEMultipart message with subject: "{subject_prefix} {run_date}".
        4. Attach HTML as MIMEText("...", "html").
        5. Connect to SMTP, starttls, login, send.

        Uses asyncio.to_thread() to wrap the synchronous smtplib calls.
        """
        self.logger.info("Preparing email for %s", run_date)

        # Combine reports
        combined_md = f"{general_report}\n\n---\n\n{specific_report}"

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
