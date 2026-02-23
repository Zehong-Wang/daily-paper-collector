import json
import logging
import sqlite3


class PaperStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._init_db()

    def _init_db(self):
        conn = self._get_conn()
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS papers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    arxiv_id TEXT UNIQUE NOT NULL,
                    title TEXT NOT NULL,
                    authors TEXT NOT NULL,
                    abstract TEXT NOT NULL,
                    categories TEXT NOT NULL,
                    published_date DATE NOT NULL,
                    pdf_url TEXT,
                    ar5iv_url TEXT,
                    embedding BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS interests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    type TEXT NOT NULL,
                    value TEXT NOT NULL,
                    description TEXT,
                    embedding BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS matches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    paper_id INTEGER REFERENCES papers(id),
                    run_date DATE NOT NULL,
                    embedding_score REAL,
                    llm_score REAL,
                    llm_reason TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(paper_id, run_date)
                );

                CREATE TABLE IF NOT EXISTS summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    paper_id INTEGER REFERENCES papers(id),
                    summary_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    llm_provider TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS daily_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_date DATE NOT NULL,
                    general_report TEXT,
                    specific_report TEXT,
                    general_report_zh TEXT,
                    specific_report_zh TEXT,
                    paper_count INTEGER,
                    matched_count INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            # Migrate existing tables: add Chinese report columns if missing
            self._migrate_add_column(conn, "daily_reports", "general_report_zh", "TEXT")
            self._migrate_add_column(conn, "daily_reports", "specific_report_zh", "TEXT")
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _migrate_add_column(conn, table: str, column: str, col_type: str):
        """Add a column to an existing table if it doesn't already exist."""
        cursor = conn.execute(f"PRAGMA table_info({table})")
        existing = {row[1] for row in cursor.fetchall()}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # --- Paper CRUD ---

    def save_papers(self, papers: list[dict]) -> list[dict]:
        """Insert papers, skip duplicates by arxiv_id. Return only newly inserted papers."""
        conn = self._get_conn()
        try:
            new_papers = []
            for paper in papers:
                authors_json = json.dumps(paper["authors"])
                categories_json = json.dumps(paper["categories"])
                cursor = conn.execute(
                    """INSERT OR IGNORE INTO papers
                       (arxiv_id, title, authors, abstract, categories,
                        published_date, pdf_url, ar5iv_url)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        paper["arxiv_id"],
                        paper["title"],
                        authors_json,
                        paper["abstract"],
                        categories_json,
                        paper["published_date"],
                        paper.get("pdf_url"),
                        paper.get("ar5iv_url"),
                    ),
                )
                if cursor.rowcount > 0:
                    paper_with_id = {**paper, "id": cursor.lastrowid}
                    new_papers.append(paper_with_id)
            conn.commit()
            return new_papers
        finally:
            conn.close()

    def get_paper_by_arxiv_id(self, arxiv_id: str) -> dict | None:
        """Return a single paper dict or None. Deserialize authors/categories from JSON."""
        conn = self._get_conn()
        try:
            row = conn.execute("SELECT * FROM papers WHERE arxiv_id = ?", (arxiv_id,)).fetchone()
            if row is None:
                return None
            return self._row_to_paper(row)
        finally:
            conn.close()

    def get_papers_by_date(self, date: str) -> list[dict]:
        """Return all papers with published_date == date."""
        conn = self._get_conn()
        try:
            rows = conn.execute("SELECT * FROM papers WHERE published_date = ?", (date,)).fetchall()
            return [self._row_to_paper(row) for row in rows]
        finally:
            conn.close()

    def search_papers(self, query: str, limit: int = 50) -> list[dict]:
        """Search papers where title or abstract LIKE '%query%'."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """SELECT * FROM papers
                   WHERE title LIKE ? OR abstract LIKE ?
                   LIMIT ?""",
                (f"%{query}%", f"%{query}%", limit),
            ).fetchall()
            return [self._row_to_paper(row) for row in rows]
        finally:
            conn.close()

    def update_paper_embedding(self, paper_id: int, embedding: bytes):
        """Update the embedding BLOB for a given paper id."""
        conn = self._get_conn()
        try:
            conn.execute("UPDATE papers SET embedding = ? WHERE id = ?", (embedding, paper_id))
            conn.commit()
        finally:
            conn.close()

    def get_papers_without_embeddings(self) -> list[dict]:
        """Return all papers where embedding IS NULL."""
        conn = self._get_conn()
        try:
            rows = conn.execute("SELECT * FROM papers WHERE embedding IS NULL").fetchall()
            return [self._row_to_paper(row) for row in rows]
        finally:
            conn.close()

    def get_papers_with_embeddings(self) -> list[dict]:
        """Return all papers where embedding IS NOT NULL. Include the embedding bytes."""
        conn = self._get_conn()
        try:
            rows = conn.execute("SELECT * FROM papers WHERE embedding IS NOT NULL").fetchall()
            return [self._row_to_paper(row) for row in rows]
        finally:
            conn.close()

    def get_papers_by_date_with_embeddings(self, date: str) -> list[dict]:
        """Return papers for a given date that have embeddings computed."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """SELECT * FROM papers
                   WHERE published_date = ? AND embedding IS NOT NULL""",
                (date,),
            ).fetchall()
            return [self._row_to_paper(row) for row in rows]
        finally:
            conn.close()

    def get_papers_in_date_range_with_embeddings(
        self, start_date: str, end_date: str
    ) -> list[dict]:
        """Return papers with published_date between start_date and end_date (inclusive)
        that have embeddings computed."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """SELECT * FROM papers
                   WHERE published_date >= ? AND published_date <= ?
                     AND embedding IS NOT NULL""",
                (start_date, end_date),
            ).fetchall()
            return [self._row_to_paper(row) for row in rows]
        finally:
            conn.close()

    def get_papers_by_ids_with_embeddings(self, paper_ids: list[int]) -> list[dict]:
        """Return papers with the given IDs that have embeddings computed."""
        if not paper_ids:
            return []
        conn = self._get_conn()
        try:
            placeholders = ",".join("?" * len(paper_ids))
            rows = conn.execute(
                f"""SELECT * FROM papers
                   WHERE id IN ({placeholders}) AND embedding IS NOT NULL""",
                paper_ids,
            ).fetchall()
            return [self._row_to_paper(row) for row in rows]
        finally:
            conn.close()

    def _row_to_paper(self, row: sqlite3.Row) -> dict:
        """Convert a sqlite3.Row to a paper dict with deserialized JSON fields."""
        d = dict(row)
        d["authors"] = json.loads(d["authors"])
        d["categories"] = json.loads(d["categories"])
        return d

    # --- Interest CRUD ---

    def save_interest(self, type: str, value: str, description: str = None) -> int:
        """Insert an interest. Return the new id."""
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                """INSERT INTO interests (type, value, description)
                   VALUES (?, ?, ?)""",
                (type, value, description),
            )
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def get_all_interests(self) -> list[dict]:
        """Return all interests."""
        conn = self._get_conn()
        try:
            rows = conn.execute("SELECT * FROM interests").fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def get_interest_by_id(self, interest_id: int) -> dict | None:
        """Return a single interest or None."""
        conn = self._get_conn()
        try:
            row = conn.execute("SELECT * FROM interests WHERE id = ?", (interest_id,)).fetchone()
            if row is None:
                return None
            return dict(row)
        finally:
            conn.close()

    def update_interest(self, interest_id: int, value: str = None, description: str = None):
        """Update fields of an interest. Only update provided (non-None) fields."""
        conn = self._get_conn()
        try:
            if value is not None:
                conn.execute(
                    "UPDATE interests SET value = ? WHERE id = ?",
                    (value, interest_id),
                )
            if description is not None:
                conn.execute(
                    "UPDATE interests SET description = ? WHERE id = ?",
                    (description, interest_id),
                )
            conn.commit()
        finally:
            conn.close()

    def delete_interest(self, interest_id: int):
        """Delete an interest by id."""
        conn = self._get_conn()
        try:
            conn.execute("DELETE FROM interests WHERE id = ?", (interest_id,))
            conn.commit()
        finally:
            conn.close()

    def update_interest_embedding(self, interest_id: int, embedding: bytes):
        """Update the embedding BLOB for a given interest id."""
        conn = self._get_conn()
        try:
            conn.execute(
                "UPDATE interests SET embedding = ? WHERE id = ?",
                (embedding, interest_id),
            )
            conn.commit()
        finally:
            conn.close()

    def get_interests_with_embeddings(self) -> list[dict]:
        """Return interests where embedding IS NOT NULL."""
        conn = self._get_conn()
        try:
            rows = conn.execute("SELECT * FROM interests WHERE embedding IS NOT NULL").fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    # --- Match CRUD ---

    def save_match(
        self,
        paper_id: int,
        run_date: str,
        embedding_score: float,
        llm_score: float = None,
        llm_reason: str = None,
    ) -> int:
        """Insert or update a match record. Return the row id."""
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                """INSERT INTO matches
                   (paper_id, run_date, embedding_score, llm_score, llm_reason)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(paper_id, run_date) DO UPDATE SET
                       embedding_score = excluded.embedding_score,
                       llm_score = excluded.llm_score,
                       llm_reason = excluded.llm_reason""",
                (paper_id, run_date, embedding_score, llm_score, llm_reason),
            )
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def get_matches_by_date(self, run_date: str) -> list[dict]:
        """Return all matches for a given run_date, joined with paper info.
        Order by llm_score DESC (nulls last), then embedding_score DESC."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """SELECT m.*, p.title, p.arxiv_id, p.abstract, p.authors,
                          p.categories, p.pdf_url
                   FROM matches m
                   JOIN papers p ON m.paper_id = p.id
                   WHERE m.run_date = ?
                   ORDER BY
                       CASE WHEN m.llm_score IS NULL THEN 1 ELSE 0 END,
                       m.llm_score DESC,
                       m.embedding_score DESC""",
                (run_date,),
            ).fetchall()
            results = []
            for row in rows:
                d = dict(row)
                d["authors"] = json.loads(d["authors"])
                d["categories"] = json.loads(d["categories"])
                results.append(d)
            return results
        finally:
            conn.close()

    # --- Summary CRUD ---

    def save_summary(
        self,
        paper_id: int,
        summary_type: str,
        content: str,
        llm_provider: str = None,
    ) -> int:
        """Insert a summary. Return new id."""
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                """INSERT INTO summaries (paper_id, summary_type, content, llm_provider)
                   VALUES (?, ?, ?, ?)""",
                (paper_id, summary_type, content, llm_provider),
            )
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def get_summary(self, paper_id: int, summary_type: str) -> dict | None:
        """Return cached summary for a paper+type, or None if not cached."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                """SELECT * FROM summaries
                   WHERE paper_id = ? AND summary_type = ?""",
                (paper_id, summary_type),
            ).fetchone()
            if row is None:
                return None
            return dict(row)
        finally:
            conn.close()

    # --- Report CRUD ---

    def save_report(
        self,
        run_date: str,
        general_report: str,
        specific_report: str,
        paper_count: int,
        matched_count: int,
        general_report_zh: str = None,
        specific_report_zh: str = None,
    ) -> int:
        """Insert a daily report record. Return new id."""
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                """INSERT INTO daily_reports
                   (run_date, general_report, specific_report,
                    general_report_zh, specific_report_zh,
                    paper_count, matched_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_date,
                    general_report,
                    specific_report,
                    general_report_zh,
                    specific_report_zh,
                    paper_count,
                    matched_count,
                ),
            )
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def get_report_by_date(self, run_date: str) -> dict | None:
        """Return the latest report for a date, or None."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM daily_reports WHERE run_date = ? ORDER BY id DESC LIMIT 1",
                (run_date,),
            ).fetchone()
            if row is None:
                return None
            return dict(row)
        finally:
            conn.close()

    def get_all_report_dates(self) -> list[str]:
        """Return all run_dates that have reports, sorted descending."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT DISTINCT run_date FROM daily_reports ORDER BY run_date DESC"
            ).fetchall()
            return [row["run_date"] for row in rows]
        finally:
            conn.close()
