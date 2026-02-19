"""Tests for PipelineScheduler (Phase 11, Step 11.1)."""

from unittest.mock import patch, MagicMock, PropertyMock

from apscheduler.triggers.cron import CronTrigger

from src.scheduler.scheduler import PipelineScheduler


class TestPipelineSchedulerInit:
    """Test PipelineScheduler initialization."""

    def test_init_stores_config(self):
        config = {"scheduler": {"cron": "30 9 * * *"}}
        scheduler = PipelineScheduler(config)
        assert scheduler.config is config

    def test_init_creates_blocking_scheduler(self):
        config = {"scheduler": {"cron": "0 8 * * *"}}
        scheduler = PipelineScheduler(config)
        assert scheduler.scheduler is not None


class TestPipelineSchedulerStart:
    """Test PipelineScheduler.start() method."""

    def test_start_adds_job_with_cron_trigger(self):
        config = {"scheduler": {"cron": "30 9 * * *"}}
        scheduler = PipelineScheduler(config)

        with patch.object(scheduler.scheduler, "add_job") as mock_add_job, \
             patch.object(scheduler.scheduler, "start"):
            scheduler.start()

            mock_add_job.assert_called_once()
            args, _kwargs = mock_add_job.call_args
            assert args[0] == scheduler._run_pipeline
            trigger = args[1]
            assert isinstance(trigger, CronTrigger)

    def test_start_parses_cron_fields_correctly(self):
        config = {"scheduler": {"cron": "30 9 * * *"}}
        scheduler = PipelineScheduler(config)

        with patch.object(scheduler.scheduler, "add_job") as mock_add_job, \
             patch.object(scheduler.scheduler, "start"):
            scheduler.start()

            trigger = mock_add_job.call_args[0][1]
            # Verify the trigger fields by checking its string representation
            trigger_str = str(trigger)
            assert "minute='30'" in trigger_str
            assert "hour='9'" in trigger_str

    def test_start_parses_daily_8am_cron(self):
        config = {"scheduler": {"cron": "0 8 * * *"}}
        scheduler = PipelineScheduler(config)

        with patch.object(scheduler.scheduler, "add_job") as mock_add_job, \
             patch.object(scheduler.scheduler, "start"):
            scheduler.start()

            trigger = mock_add_job.call_args[0][1]
            trigger_str = str(trigger)
            assert "minute='0'" in trigger_str
            assert "hour='8'" in trigger_str

    def test_start_parses_weekday_only_cron(self):
        config = {"scheduler": {"cron": "0 8 * * 1-5"}}
        scheduler = PipelineScheduler(config)

        with patch.object(scheduler.scheduler, "add_job") as mock_add_job, \
             patch.object(scheduler.scheduler, "start"):
            scheduler.start()

            trigger = mock_add_job.call_args[0][1]
            trigger_str = str(trigger)
            assert "day_of_week='1-5'" in trigger_str

    def test_start_calls_scheduler_start(self):
        config = {"scheduler": {"cron": "30 9 * * *"}}
        scheduler = PipelineScheduler(config)

        with patch.object(scheduler.scheduler, "add_job"), \
             patch.object(scheduler.scheduler, "start") as mock_start:
            scheduler.start()
            mock_start.assert_called_once()


class TestPipelineSchedulerRunPipeline:
    """Test PipelineScheduler._run_pipeline() method."""

    @patch("src.scheduler.scheduler.asyncio.run")
    @patch("src.scheduler.scheduler.DailyPipeline", create=True)
    def test_run_pipeline_creates_and_runs_pipeline(self, mock_pipeline_cls, mock_asyncio_run):
        config = {"scheduler": {"cron": "30 9 * * *"}}
        scheduler = PipelineScheduler(config)

        mock_pipeline_instance = MagicMock()
        mock_pipeline_cls.return_value = mock_pipeline_instance
        mock_asyncio_run.return_value = {
            "date": "2026-02-19",
            "papers_fetched": 10,
            "new_papers": 8,
            "matches": 3,
            "email_sent": True,
        }

        with patch("src.pipeline.DailyPipeline", mock_pipeline_cls):
            scheduler._run_pipeline()

        mock_asyncio_run.assert_called_once_with(mock_pipeline_instance.run())

    @patch("src.scheduler.scheduler.asyncio.run")
    def test_run_pipeline_passes_config_to_pipeline(self, mock_asyncio_run):
        config = {
            "scheduler": {"cron": "30 9 * * *"},
            "database": {"path": "/tmp/test.db"},
        }
        scheduler = PipelineScheduler(config)

        with patch("src.pipeline.DailyPipeline") as mock_pipeline_cls:
            mock_pipeline_instance = MagicMock()
            mock_pipeline_cls.return_value = mock_pipeline_instance
            mock_asyncio_run.return_value = {"date": "2026-02-19"}

            scheduler._run_pipeline()

            mock_pipeline_cls.assert_called_once_with(config)
