import asyncio
import logging

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger


class PipelineScheduler:
    def __init__(self, config: dict):
        self.config = config
        self.scheduler = BlockingScheduler()
        self.logger = logging.getLogger(__name__)

    def start(self):
        """Start the scheduler with the configured cron expression.

        Parse cron string from config["scheduler"]["cron"] (format: "M H * * *").
        Split into fields: minute, hour, day, month, day_of_week.
        Add job using CronTrigger.
        Call self.scheduler.start() (blocks forever).
        """
        cron = self.config["scheduler"]["cron"]
        parts = cron.split()
        trigger = CronTrigger(
            minute=parts[0],
            hour=parts[1],
            day=parts[2],
            month=parts[3],
            day_of_week=parts[4],
        )
        self.scheduler.add_job(self._run_pipeline, trigger)
        self.logger.info("Scheduler started with cron: %s", cron)
        self.scheduler.start()

    def _run_pipeline(self):
        """Create and run the DailyPipeline inside an asyncio event loop."""
        from src.pipeline import DailyPipeline

        self.logger.info("Scheduled pipeline run triggered")
        pipeline = DailyPipeline(self.config)
        result = asyncio.run(pipeline.run())
        self.logger.info("Pipeline completed: %s", result)
