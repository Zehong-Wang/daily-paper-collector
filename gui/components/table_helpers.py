from __future__ import annotations


def truncate_authors(authors: list | str, max_count: int = 3) -> str:
    """Truncate author list to max_count names + 'et al.' if needed."""
    if isinstance(authors, str):
        return authors
    if not authors:
        return ""
    if len(authors) <= max_count:
        return ", ".join(authors)
    return ", ".join(authors[:max_count]) + " et al."


def truncate_text(text: str, max_len: int = 80) -> str:
    """Truncate text to max_len characters + '...' if needed."""
    if not text or len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + "..."


def get_primary_category(categories: list | str) -> str:
    """Return the first category from the list."""
    if isinstance(categories, list):
        return categories[0] if categories else "unknown"
    return str(categories) if categories else "unknown"
