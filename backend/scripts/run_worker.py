"""CLI entry point for the durable ingestion worker."""

from __future__ import annotations

import argparse
import logging

from app.core.config import get_settings
from app.indexing.startup import validate_index_startup
from app.jobs.runner import JobWorker

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the worker CLI argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process at most one queued job and exit.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the worker until interrupted, or once when requested."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    settings = get_settings()
    validate_index_startup(settings)
    worker = JobWorker(settings=settings)
    if args.once:
        processed = worker.run_once()
        logger.info("worker run_once finished", extra={"processed": processed})
        return 0
    try:
        worker.run_forever()
    except KeyboardInterrupt:
        logger.info("worker interrupted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
