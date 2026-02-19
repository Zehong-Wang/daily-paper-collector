import argparse
import asyncio
import logging


def main():
    parser = argparse.ArgumentParser(description="Daily Paper Collector")
    parser.add_argument(
        "--mode",
        choices=["scheduler", "run"],
        default="run",
        help="'scheduler' starts the cron scheduler; 'run' executes once.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config file (default: config/config.yaml relative to project root).",
    )
    args = parser.parse_args()

    from src.config import load_config, setup_logging

    setup_logging()
    config = load_config(args.config)

    logger = logging.getLogger(__name__)

    if args.mode == "scheduler":
        from src.scheduler.scheduler import PipelineScheduler

        logger.info("Starting in scheduler mode")
        scheduler = PipelineScheduler(config)
        scheduler.start()
    elif args.mode == "run":
        from src.pipeline import DailyPipeline

        logger.info("Starting single pipeline run")
        pipeline = DailyPipeline(config)
        result = asyncio.run(pipeline.run())
        logger.info("Pipeline completed: %s", result)


if __name__ == "__main__":
    main()
