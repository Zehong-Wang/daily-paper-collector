"""Standalone entry point for GitHub Actions / CI/CD."""

import asyncio
import logging
import sys
import os

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config, setup_logging
from src.pipeline import DailyPipeline


def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    config = load_config()
    pipeline = DailyPipeline(config)
    result = asyncio.run(pipeline.run())
    logger.info("Pipeline completed: %s", result)

    if result["new_papers"] == 0:
        logger.warning("No new papers fetched.")


if __name__ == "__main__":
    main()
