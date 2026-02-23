from datetime import date

import pytest

from src.store.database import PaperStore


@pytest.fixture
def store(tmp_path):
    """Create a PaperStore with a temporary database."""
    return PaperStore(str(tmp_path / "test.db"))


@pytest.fixture
def sample_papers():
    """Return 3 sample paper dicts for testing."""
    today = date.today().isoformat()
    return [
        {
            "arxiv_id": "2501.00001",
            "title": "Transformer Architectures for NLP",
            "authors": ["Alice Smith", "Bob Jones"],
            "abstract": "We present a novel transformer architecture for natural language processing.",
            "categories": ["cs.CL", "cs.AI"],
            "published_date": today,
            "pdf_url": "https://arxiv.org/pdf/2501.00001",
            "ar5iv_url": "https://ar5iv.labs.arxiv.org/html/2501.00001",
        },
        {
            "arxiv_id": "2501.00002",
            "title": "Deep Reinforcement Learning Survey",
            "authors": ["Charlie Brown"],
            "abstract": "A comprehensive survey of deep reinforcement learning methods.",
            "categories": ["cs.LG"],
            "published_date": today,
            "pdf_url": "https://arxiv.org/pdf/2501.00002",
            "ar5iv_url": "https://ar5iv.labs.arxiv.org/html/2501.00002",
        },
        {
            "arxiv_id": "2501.00003",
            "title": "Computer Vision with Diffusion Models",
            "authors": ["Diana Prince", "Eve Adams"],
            "abstract": "Applying diffusion models to computer vision tasks.",
            "categories": ["cs.CV", "cs.LG"],
            "published_date": today,
            "pdf_url": "https://arxiv.org/pdf/2501.00003",
            "ar5iv_url": "https://ar5iv.labs.arxiv.org/html/2501.00003",
        },
    ]


# --- Step 1.1: Schema Initialization ---


class TestSchemaInit:
    def test_db_file_created(self, tmp_path):
        db_path = tmp_path / "test.db"
        PaperStore(str(db_path))
        assert db_path.exists()

    def test_all_tables_exist(self, store):
        conn = store._get_conn()
        try:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
            table_names = {row["name"] for row in rows}
            expected = {"papers", "interests", "matches", "summaries", "daily_reports"}
            assert expected.issubset(table_names)
        finally:
            conn.close()

    def test_idempotent_init(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        PaperStore(db_path)
        store2 = PaperStore(db_path)
        # Both should work without error; verify tables still exist
        conn = store2._get_conn()
        try:
            rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            assert len(rows) >= 5
        finally:
            conn.close()


# --- Step 1.2: Paper CRUD ---


class TestPaperCRUD:
    def test_save_papers_returns_new(self, store, sample_papers):
        result = store.save_papers(sample_papers)
        assert len(result) == 3
        for paper in result:
            assert "id" in paper
            assert paper["id"] > 0

    def test_save_papers_skips_duplicates(self, store, sample_papers):
        store.save_papers(sample_papers)
        result = store.save_papers(sample_papers)
        assert len(result) == 0

    def test_get_paper_by_arxiv_id_found(self, store, sample_papers):
        store.save_papers(sample_papers)
        paper = store.get_paper_by_arxiv_id("2501.00001")
        assert paper is not None
        assert paper["title"] == "Transformer Architectures for NLP"
        assert paper["authors"] == ["Alice Smith", "Bob Jones"]
        assert paper["categories"] == ["cs.CL", "cs.AI"]

    def test_get_paper_by_arxiv_id_not_found(self, store):
        paper = store.get_paper_by_arxiv_id("9999.99999")
        assert paper is None

    def test_get_papers_by_date(self, store, sample_papers):
        store.save_papers(sample_papers)
        today = date.today().isoformat()
        papers = store.get_papers_by_date(today)
        assert len(papers) == 3

    def test_get_papers_by_date_no_match(self, store, sample_papers):
        store.save_papers(sample_papers)
        papers = store.get_papers_by_date("2000-01-01")
        assert len(papers) == 0

    def test_search_papers(self, store, sample_papers):
        store.save_papers(sample_papers)
        results = store.search_papers("transformer")
        assert len(results) >= 1
        assert any("Transformer" in p["title"] for p in results)

    def test_search_papers_by_abstract(self, store, sample_papers):
        store.save_papers(sample_papers)
        results = store.search_papers("reinforcement learning")
        assert len(results) >= 1

    def test_search_papers_no_match(self, store, sample_papers):
        store.save_papers(sample_papers)
        results = store.search_papers("quantum computing xyz123")
        assert len(results) == 0

    def test_update_and_get_paper_embedding(self, store, sample_papers):
        new_papers = store.save_papers(sample_papers)
        paper_id = new_papers[0]["id"]
        fake_blob = b"fake_embedding_blob"
        store.update_paper_embedding(paper_id, fake_blob)

        with_embeddings = store.get_papers_with_embeddings()
        assert len(with_embeddings) == 1
        assert with_embeddings[0]["embedding"] == fake_blob

    def test_get_papers_without_embeddings(self, store, sample_papers):
        new_papers = store.save_papers(sample_papers)
        store.update_paper_embedding(new_papers[0]["id"], b"blob")

        without = store.get_papers_without_embeddings()
        assert len(without) == 2

    def test_get_papers_by_date_with_embeddings(self, store, sample_papers):
        today = date.today().isoformat()
        new_papers = store.save_papers(sample_papers)
        store.update_paper_embedding(new_papers[0]["id"], b"blob")

        result = store.get_papers_by_date_with_embeddings(today)
        assert len(result) == 1
        assert result[0]["arxiv_id"] == "2501.00001"

    def test_get_papers_by_date_with_embeddings_different_date(self, store):
        """Paper with embedding but on a different date should NOT be returned."""
        today = date.today().isoformat()
        paper_today = {
            "arxiv_id": "2501.11111",
            "title": "Paper Today",
            "authors": ["A"],
            "abstract": "Today's abstract",
            "categories": ["cs.AI"],
            "published_date": today,
            "pdf_url": "https://arxiv.org/pdf/2501.11111",
            "ar5iv_url": "https://ar5iv.labs.arxiv.org/html/2501.11111",
        }
        paper_old = {
            "arxiv_id": "2501.22222",
            "title": "Paper Old",
            "authors": ["B"],
            "abstract": "Old abstract",
            "categories": ["cs.LG"],
            "published_date": "2020-01-01",
            "pdf_url": "https://arxiv.org/pdf/2501.22222",
            "ar5iv_url": "https://ar5iv.labs.arxiv.org/html/2501.22222",
        }
        new = store.save_papers([paper_today, paper_old])
        # Give both embeddings
        for p in new:
            store.update_paper_embedding(p["id"], b"blob")

        result = store.get_papers_by_date_with_embeddings(today)
        assert len(result) == 1
        assert result[0]["arxiv_id"] == "2501.11111"

    def test_authors_categories_deserialized(self, store, sample_papers):
        """Verify authors and categories are properly deserialized from JSON."""
        store.save_papers(sample_papers)
        paper = store.get_paper_by_arxiv_id("2501.00001")
        assert isinstance(paper["authors"], list)
        assert isinstance(paper["categories"], list)
        assert paper["authors"] == ["Alice Smith", "Bob Jones"]
        assert paper["categories"] == ["cs.CL", "cs.AI"]


# --- Step 1.3: Interest CRUD ---


class TestInterestCRUD:
    def test_save_interest_keyword(self, store):
        interest_id = store.save_interest("keyword", "transformer", "attention mechanism")
        assert interest_id > 0

    def test_save_interest_paper(self, store):
        interest_id = store.save_interest("paper", "2501.12345")
        assert interest_id > 0

    def test_get_all_interests(self, store):
        store.save_interest("keyword", "transformer")
        store.save_interest("paper", "2501.12345")
        interests = store.get_all_interests()
        assert len(interests) == 2

    def test_get_interest_by_id(self, store):
        iid = store.save_interest("keyword", "transformer", "attention mechanism")
        interest = store.get_interest_by_id(iid)
        assert interest is not None
        assert interest["type"] == "keyword"
        assert interest["value"] == "transformer"
        assert interest["description"] == "attention mechanism"

    def test_get_interest_by_id_not_found(self, store):
        assert store.get_interest_by_id(9999) is None

    def test_update_interest_value(self, store):
        iid = store.save_interest("keyword", "transformer")
        store.update_interest(iid, value="attention mechanisms")
        updated = store.get_interest_by_id(iid)
        assert updated["value"] == "attention mechanisms"

    def test_update_interest_description(self, store):
        iid = store.save_interest("keyword", "transformer")
        store.update_interest(iid, description="self-attention models")
        updated = store.get_interest_by_id(iid)
        assert updated["description"] == "self-attention models"

    def test_update_interest_both_fields(self, store):
        iid = store.save_interest("keyword", "old_value")
        store.update_interest(iid, value="new_value", description="new_desc")
        updated = store.get_interest_by_id(iid)
        assert updated["value"] == "new_value"
        assert updated["description"] == "new_desc"

    def test_delete_interest(self, store):
        iid1 = store.save_interest("keyword", "transformer")
        iid2 = store.save_interest("paper", "2501.12345")
        store.delete_interest(iid1)
        interests = store.get_all_interests()
        assert len(interests) == 1
        assert interests[0]["id"] == iid2

    def test_update_interest_embedding(self, store):
        iid = store.save_interest("keyword", "transformer")
        store.update_interest_embedding(iid, b"interest_embedding_blob")
        with_emb = store.get_interests_with_embeddings()
        assert len(with_emb) == 1
        assert with_emb[0]["embedding"] == b"interest_embedding_blob"

    def test_get_interests_with_embeddings_filters_null(self, store):
        store.save_interest("keyword", "transformer")
        iid2 = store.save_interest("paper", "2501.12345")
        store.update_interest_embedding(iid2, b"blob")
        with_emb = store.get_interests_with_embeddings()
        assert len(with_emb) == 1
        assert with_emb[0]["value"] == "2501.12345"


# --- Step 1.4: Match, Summary, Report CRUD ---


class TestMatchCRUD:
    def test_save_and_get_match(self, store, sample_papers):
        new_papers = store.save_papers(sample_papers)
        paper_id = new_papers[0]["id"]
        run_date = date.today().isoformat()

        match_id = store.save_match(paper_id, run_date, 0.85, 8.5, "Highly relevant")
        assert match_id > 0

        matches = store.get_matches_by_date(run_date)
        assert len(matches) == 1
        assert matches[0]["paper_id"] == paper_id
        assert matches[0]["embedding_score"] == 0.85
        assert matches[0]["llm_score"] == 8.5
        assert matches[0]["llm_reason"] == "Highly relevant"
        # Joined paper info
        assert matches[0]["title"] == "Transformer Architectures for NLP"
        assert matches[0]["arxiv_id"] == "2501.00001"
        assert isinstance(matches[0]["authors"], list)

    def test_matches_ordered_by_scores(self, store, sample_papers):
        new_papers = store.save_papers(sample_papers)
        run_date = date.today().isoformat()

        store.save_match(new_papers[0]["id"], run_date, 0.9, 7.0, "Good")
        store.save_match(new_papers[1]["id"], run_date, 0.8, 9.0, "Great")
        store.save_match(new_papers[2]["id"], run_date, 0.7, None, None)

        matches = store.get_matches_by_date(run_date)
        assert len(matches) == 3
        # LLM-scored papers come first (nulls last), then by llm_score DESC
        assert matches[0]["llm_score"] == 9.0
        assert matches[1]["llm_score"] == 7.0
        assert matches[2]["llm_score"] is None

    def test_get_matches_no_results(self, store):
        matches = store.get_matches_by_date("2000-01-01")
        assert len(matches) == 0


class TestSummaryCRUD:
    def test_save_and_get_summary(self, store, sample_papers):
        new_papers = store.save_papers(sample_papers)
        paper_id = new_papers[0]["id"]

        summary_id = store.save_summary(paper_id, "brief", "A brief summary.", "openai")
        assert summary_id > 0

        summary = store.get_summary(paper_id, "brief")
        assert summary is not None
        assert summary["content"] == "A brief summary."
        assert summary["llm_provider"] == "openai"

    def test_get_summary_not_found(self, store, sample_papers):
        new_papers = store.save_papers(sample_papers)
        paper_id = new_papers[0]["id"]

        store.save_summary(paper_id, "brief", "Brief.")
        # Detailed not saved
        assert store.get_summary(paper_id, "detailed") is None

    def test_get_summary_no_paper(self, store):
        assert store.get_summary(9999, "brief") is None


class TestReportCRUD:
    def test_save_and_get_report(self, store):
        run_date = date.today().isoformat()
        report_id = store.save_report(run_date, "# General Report", "## Specific Report", 100, 10)
        assert report_id > 0

        report = store.get_report_by_date(run_date)
        assert report is not None
        assert report["general_report"] == "# General Report"
        assert report["specific_report"] == "## Specific Report"
        assert report["paper_count"] == 100
        assert report["matched_count"] == 10

    def test_get_report_not_found(self, store):
        assert store.get_report_by_date("2000-01-01") is None

    def test_get_all_report_dates(self, store):
        store.save_report("2025-01-15", "gen1", "spec1", 50, 5)
        store.save_report("2025-01-16", "gen2", "spec2", 60, 6)
        store.save_report("2025-01-14", "gen3", "spec3", 40, 4)

        dates = store.get_all_report_dates()
        assert len(dates) == 3
        # Should be sorted descending
        assert dates[0] == "2025-01-16"
        assert dates[1] == "2025-01-15"
        assert dates[2] == "2025-01-14"

    def test_get_all_report_dates_empty(self, store):
        dates = store.get_all_report_dates()
        assert dates == []


# --- Range Report CRUD ---


class TestRangeReportCRUD:
    def test_save_report_with_report_type(self, store):
        report_id = store.save_report(
            "2026-02-20~2026-02-22",
            "# General",
            "## Specific",
            150,
            20,
            report_type="3day",
        )
        assert report_id > 0
        report = store.get_report_by_id(report_id)
        assert report["report_type"] == "3day"
        assert report["run_date"] == "2026-02-20~2026-02-22"

    def test_save_report_default_type_is_daily(self, store):
        report_id = store.save_report("2026-02-22", "# Gen", "## Spec", 100, 10)
        report = store.get_report_by_id(report_id)
        assert report["report_type"] == "daily"

    def test_get_all_report_entries(self, store):
        store.save_report("2026-02-22", "gen1", "spec1", 50, 5)
        store.save_report(
            "2026-02-20~2026-02-22", "gen2", "spec2", 150, 20, report_type="3day"
        )
        store.save_report(
            "2026-02-16~2026-02-22", "gen3", "spec3", 400, 30, report_type="1week"
        )

        entries = store.get_all_report_entries()
        assert len(entries) == 3
        # Should include id, run_date, report_type, paper_count, matched_count, created_at
        for e in entries:
            assert "id" in e
            assert "run_date" in e
            assert "report_type" in e
            assert "paper_count" in e
            assert "matched_count" in e
            assert "created_at" in e
        # Should NOT include full report text
        assert "general_report" not in entries[0]

    def test_get_all_report_entries_sorted_by_created_at_desc(self, store):
        id1 = store.save_report("2026-02-20", "g1", "s1", 50, 5)
        id2 = store.save_report("2026-02-21", "g2", "s2", 60, 6)
        id3 = store.save_report("2026-02-22", "g3", "s3", 70, 7)

        entries = store.get_all_report_entries()
        ids = [e["id"] for e in entries]
        assert ids == [id3, id2, id1]

    def test_get_report_by_id(self, store):
        id1 = store.save_report("2026-02-22", "# Gen1", "## Spec1", 100, 10)
        id2 = store.save_report(
            "2026-02-20~2026-02-22", "# Gen2", "## Spec2", 150, 20, report_type="3day"
        )

        report1 = store.get_report_by_id(id1)
        assert report1["general_report"] == "# Gen1"
        assert report1["report_type"] == "daily"

        report2 = store.get_report_by_id(id2)
        assert report2["general_report"] == "# Gen2"
        assert report2["report_type"] == "3day"

    def test_get_report_by_id_not_found(self, store):
        assert store.get_report_by_id(9999) is None

    def test_report_type_column_migration(self, tmp_path):
        """Verify that report_type column is added via migration."""
        store = PaperStore(str(tmp_path / "test.db"))
        conn = store._get_conn()
        try:
            cursor = conn.execute("PRAGMA table_info(daily_reports)")
            columns = {row[1] for row in cursor.fetchall()}
            assert "report_type" in columns
        finally:
            conn.close()

    def test_save_report_with_chinese_and_type(self, store):
        report_id = store.save_report(
            "2026-02-20~2026-02-22",
            "# General",
            "## Specific",
            150,
            20,
            general_report_zh="# 综合报告",
            specific_report_zh="## 个性化推荐",
            report_type="1week",
        )
        report = store.get_report_by_id(report_id)
        assert report["general_report_zh"] == "# 综合报告"
        assert report["specific_report_zh"] == "## 个性化推荐"
        assert report["report_type"] == "1week"

    def test_get_all_report_entries_empty(self, store):
        entries = store.get_all_report_entries()
        assert entries == []
