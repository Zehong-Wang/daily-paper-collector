"""Tests for CLI entry point (Phase 11, Step 11.2)."""

from unittest.mock import patch, MagicMock


class TestMainRunMode:
    """Test main() in 'run' mode."""

    @patch("src.main.asyncio.run")
    @patch("src.pipeline.DailyPipeline")
    @patch("src.config.load_config")
    @patch("src.config.setup_logging")
    @patch("sys.argv", ["main", "--mode", "run"])
    def test_run_mode_creates_and_runs_pipeline(
        self, mock_setup_logging, mock_load_config, mock_pipeline_cls, mock_asyncio_run
    ):
        from src.main import main

        mock_config = {"database": {"path": "/tmp/test.db"}}
        mock_load_config.return_value = mock_config
        mock_pipeline_instance = MagicMock()
        mock_pipeline_cls.return_value = mock_pipeline_instance
        mock_asyncio_run.return_value = {
            "date": "2026-02-19",
            "papers_fetched": 10,
            "new_papers": 8,
            "matches": 3,
            "email_sent": True,
        }

        main()

        mock_setup_logging.assert_called_once()
        mock_load_config.assert_called_once_with(None)
        mock_asyncio_run.assert_called_once()

    @patch("src.main.asyncio.run")
    @patch("src.pipeline.DailyPipeline")
    @patch("src.config.load_config")
    @patch("src.config.setup_logging")
    @patch("sys.argv", ["main", "--mode", "run", "--config", "custom/config.yaml"])
    def test_run_mode_with_custom_config(
        self, mock_setup_logging, mock_load_config, mock_pipeline_cls, mock_asyncio_run
    ):
        from src.main import main

        mock_load_config.return_value = {"database": {"path": "/tmp/test.db"}}
        mock_pipeline_cls.return_value = MagicMock()
        mock_asyncio_run.return_value = {"date": "2026-02-19"}

        main()

        mock_load_config.assert_called_once_with("custom/config.yaml")

    @patch("src.main.asyncio.run")
    @patch("src.pipeline.DailyPipeline")
    @patch("src.config.load_config")
    @patch("src.config.setup_logging")
    @patch("sys.argv", ["main"])
    def test_default_mode_is_run(
        self, mock_setup_logging, mock_load_config, mock_pipeline_cls, mock_asyncio_run
    ):
        from src.main import main

        mock_load_config.return_value = {"database": {"path": "/tmp/test.db"}}
        mock_pipeline_cls.return_value = MagicMock()
        mock_asyncio_run.return_value = {"date": "2026-02-19"}

        main()

        # asyncio.run is called (run mode), not PipelineScheduler.start
        mock_asyncio_run.assert_called_once()


class TestMainSchedulerMode:
    """Test main() in 'scheduler' mode."""

    @patch("src.scheduler.scheduler.PipelineScheduler")
    @patch("src.config.load_config")
    @patch("src.config.setup_logging")
    @patch("sys.argv", ["main", "--mode", "scheduler"])
    def test_scheduler_mode_creates_and_starts_scheduler(
        self, mock_setup_logging, mock_load_config, mock_scheduler_cls
    ):
        from src.main import main

        mock_config = {"scheduler": {"cron": "0 8 * * *"}}
        mock_load_config.return_value = mock_config
        mock_scheduler_instance = MagicMock()
        mock_scheduler_cls.return_value = mock_scheduler_instance

        main()

        mock_setup_logging.assert_called_once()
        mock_scheduler_instance.start.assert_called_once()

    @patch("src.scheduler.scheduler.PipelineScheduler")
    @patch("src.config.load_config")
    @patch("src.config.setup_logging")
    @patch("sys.argv", ["main", "--mode", "scheduler"])
    def test_scheduler_mode_passes_config(
        self, mock_setup_logging, mock_load_config, mock_scheduler_cls
    ):
        from src.main import main

        mock_config = {"scheduler": {"cron": "30 9 * * *"}}
        mock_load_config.return_value = mock_config
        mock_scheduler_instance = MagicMock()
        mock_scheduler_cls.return_value = mock_scheduler_instance

        main()

        mock_scheduler_cls.assert_called_once_with(mock_config)


class TestRunPipelineScript:
    """Test scripts/run_pipeline.py."""

    @patch("scripts.run_pipeline.DailyPipeline")
    @patch("scripts.run_pipeline.load_config")
    @patch("scripts.run_pipeline.setup_logging")
    def test_run_pipeline_script_creates_and_runs(
        self, mock_setup_logging, mock_load_config, mock_pipeline_cls
    ):
        mock_config = {"database": {"path": "/tmp/test.db"}}
        mock_load_config.return_value = mock_config
        mock_pipeline_instance = MagicMock()
        mock_pipeline_cls.return_value = mock_pipeline_instance

        async def mock_run():
            return {
                "date": "2026-02-19",
                "papers_fetched": 10,
                "new_papers": 5,
                "matches": 3,
                "email_sent": False,
            }

        mock_pipeline_instance.run = mock_run

        from scripts.run_pipeline import main as run_main

        run_main()

        mock_pipeline_cls.assert_called_once_with(mock_config)

    @patch("scripts.run_pipeline.DailyPipeline")
    @patch("scripts.run_pipeline.load_config")
    @patch("scripts.run_pipeline.setup_logging")
    def test_run_pipeline_script_warns_on_no_new_papers(
        self, mock_setup_logging, mock_load_config, mock_pipeline_cls, caplog
    ):
        import logging

        mock_config = {"database": {"path": "/tmp/test.db"}}
        mock_load_config.return_value = mock_config
        mock_pipeline_instance = MagicMock()
        mock_pipeline_cls.return_value = mock_pipeline_instance

        async def mock_run():
            return {
                "date": "2026-02-19",
                "papers_fetched": 0,
                "new_papers": 0,
                "matches": 0,
                "email_sent": False,
            }

        mock_pipeline_instance.run = mock_run

        with caplog.at_level(logging.WARNING):
            from scripts.run_pipeline import main as run_main

            run_main()

        assert any("No new papers" in record.message for record in caplog.records)
