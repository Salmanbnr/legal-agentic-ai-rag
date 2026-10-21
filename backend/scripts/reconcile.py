"""Reconcile Postgres chunk ids with Pinecone vectors for one case."""

from __future__ import annotations

import argparse
import json
import logging
from uuid import UUID

from app.core.config import get_settings
from app.db.client import get_service_client
from app.indexing.reconcile import fix_reconcile, reconcile_case

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the reconciliation CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case-id",
        required=True,
        type=UUID,
        help="Case id whose namespace should be reconciled.",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Delete orphaned vectors and queue reindex jobs for missing ids.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Report or repair one case namespace without printing document text."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    settings = get_settings()
    client = get_service_client(settings)

    report = reconcile_case(client, args.case_id, settings=settings)
    payload = {
        "case_id": str(report.case_id),
        "postgres_chunk_count": report.postgres_chunk_count,
        "pinecone_vector_count": report.pinecone_vector_count,
        "orphaned_ids": report.orphaned_ids,
        "missing_ids": report.missing_ids,
    }
    if args.fix and not report.is_clean:
        fixed = fix_reconcile(
            client, args.case_id, settings=settings
        )
        payload["fixed"] = {
            "deleted_orphan_count": fixed.deleted_orphan_count,
            "requeued_document_ids": fixed.requeued_document_ids,
        }
    print(json.dumps(payload, indent=2))
    if report.is_clean or args.fix:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
