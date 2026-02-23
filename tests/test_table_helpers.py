from gui.components.table_helpers import (
    truncate_authors,
    truncate_text,
    get_primary_category,
)


class TestTruncateAuthors:
    def test_three_or_fewer_returns_full(self):
        assert truncate_authors(["Alice", "Bob", "Charlie"]) == "Alice, Bob, Charlie"

    def test_more_than_three_truncates(self):
        authors = ["Alice", "Bob", "Charlie", "Diana"]
        assert truncate_authors(authors) == "Alice, Bob, Charlie et al."

    def test_single_author(self):
        assert truncate_authors(["Alice"]) == "Alice"

    def test_string_passthrough(self):
        assert truncate_authors("Alice, Bob") == "Alice, Bob"

    def test_empty_list(self):
        assert truncate_authors([]) == ""

    def test_custom_max_count(self):
        authors = ["A", "B", "C", "D", "E"]
        assert truncate_authors(authors, max_count=2) == "A, B et al."


class TestTruncateText:
    def test_short_text_unchanged(self):
        assert truncate_text("hello world") == "hello world"

    def test_exact_length_unchanged(self):
        text = "x" * 80
        assert truncate_text(text) == text

    def test_long_text_truncated(self):
        text = "a" * 100
        result = truncate_text(text, 80)
        assert result == "a" * 80 + "..."
        assert len(result) == 83

    def test_empty_string(self):
        assert truncate_text("") == ""

    def test_custom_max_len(self):
        result = truncate_text("hello world", max_len=5)
        assert result == "hello..."


class TestGetPrimaryCategory:
    def test_list_returns_first(self):
        assert get_primary_category(["cs.AI", "cs.CL"]) == "cs.AI"

    def test_string_passthrough(self):
        assert get_primary_category("cs.AI") == "cs.AI"

    def test_empty_list_returns_unknown(self):
        assert get_primary_category([]) == "unknown"

    def test_single_item_list(self):
        assert get_primary_category(["cs.LG"]) == "cs.LG"

    def test_empty_string_returns_unknown(self):
        assert get_primary_category("") == "unknown"
