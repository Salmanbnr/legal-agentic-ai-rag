"""Create or recreate the configured Pinecone serverless index.

Pinecone is derived state: the index can always be rebuilt from Postgres.
Recreating is only needed when the index was created with an incompatible
schema, for example a document schema that rejects vector writes.
"""

from __future__ import annotations

import argparse
import logging

from app.core.config import get_settings
from app.indexing.pinecone_store import (
    create_index_if_missing,
    delete_index,
    describe_index_dimension,
)

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the index-creation CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dimension",
        type=int,
        default=None,
        help="Override the configured embedding dimension.",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete the existing index first. Requires --yes.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm the deletion that --recreate performs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Create the index if needed and print only safe index metadata."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.recreate and not args.yes:
        parser.error("--recreate deletes the index and needs --yes")

    logging.basicConfig(level=logging.INFO)
    settings = get_settings()
    name = settings.pinecone_index_name

    if args.recreate:
        deleted = delete_index(settings)
        print(f"Pinecone index {name}: {'deleted' if deleted else 'absent'}")

    status = create_index_if_missing(settings, dimension=args.dimension)
    dimension = describe_index_dimension(settings)
    print(
        f"Pinecone index {name}: {status} "
        f"(dimension={dimension}, metric={settings.pinecone_metric})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
