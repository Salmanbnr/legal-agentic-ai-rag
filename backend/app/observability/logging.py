"""Application logging configuration.

Log lines carry identifiers, counts, timings, status codes, and error classes
only. Document text, prompts, completions, tokens, credentials, and secrets are
never logged, and the formatter only renders an explicit allow-list of
structured fields so an accidental extra cannot leak case content.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

#: Logger that owns the application handlers. Every module logger is a child
#: (``app.*``) and propagates to it.
APP_LOGGER_NAME = "app"

#: Structured fields that are safe to print. New fields must be added here.
SAFE_EXTRA_KEYS: frozenset[str] = frozenset(
    {
        "attempt",
        "case_id",
        "chunk_count",
        "chunk_id",
        "claim_count",
        "conflict_count",
        "delay_seconds",
        "deleted_orphan_count",
        "deleted_vector_count",
        "dense_candidates",
        "document_count",
        "document_id",
        "duration_ms",
        "embedded_count",
        "embedding_device",
        "embedding_dimension",
        "embedding_model",
        "error_class",
        "error_code",
        "evidence_count",
        "fact_count",
        "failed_pages",
        "final_answer",
        "final_status",
        "gap_count",
        "graph_version",
        "indexed_count",
        "indexed_vector_count",
        "job_id",
        "job_type",
        "lexical_candidates",
        "llm_calls",
        "llm_calls_used",
        "low_confidence_pages",
        "material_failure_count",
        "method",
        "missing_count",
        "mode",
        "namespace",
        "node",
        "node_status",
        "normalized_pages",
        "ocr_pages",
        "orphaned_count",
        "page_count",
        "pages_empty",
        "pages_total",
        "parser_errors",
        "path",
        "pinecone_index",
        "pinecone_vector_count",
        "pipeline_version",
        "postgres_chunk_count",
        "provider_status",
        "quality_status",
        "reason_codes",
        "removed_claim_count",
        "removed_evidence_count",
        "request_id",
        "requeued_document_count",
        "retries",
        "retryable",
        "rewrite_strategy",
        "span_count",
        "stage",
        "status",
        "status_code",
        "steps",
        "subquestion_id",
        "task_type",
        "tokens",
        "tokens_used",
    }
)

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S%z"
_HANDLER_MARKER = "_legal_ai_handler"


class SafeFormatter(logging.Formatter):
    """Render the message plus allow-listed structured fields."""

    def format(self, record: logging.LogRecord) -> str:
        """Return one line with safe extras appended as ``key=value``."""
        base = super().format(record)
        parts = [
            f"{key}={record.__dict__[key]}"
            for key in sorted(SAFE_EXTRA_KEYS)
            if key in record.__dict__
        ]
        return f"{base} {' '.join(parts)}" if parts else base


def _build_formatter() -> logging.Formatter:
    """Return the shared formatter instance style."""
    return SafeFormatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)


def _remove_own_handlers(logger: logging.Logger) -> None:
    """Drop handlers installed by an earlier call so setup stays idempotent."""
    for handler in list(logger.handlers):
        if getattr(handler, _HANDLER_MARKER, False):
            logger.removeHandler(handler)
            handler.close()


def configure_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """Attach a console handler and an optional rotating file handler.

    Args:
        level: Minimum level for the ``app`` logger tree.
        log_file: Optional path for rotating file output. Parent directories
            are created when missing.
    """
    logger = logging.getLogger(APP_LOGGER_NAME)
    logger.setLevel(level.upper())
    logger.propagate = False
    _remove_own_handlers(logger)

    console = logging.StreamHandler()
    console.setFormatter(_build_formatter())
    setattr(console, _HANDLER_MARKER, True)
    logger.addHandler(console)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            path,
            maxBytes=5_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(_build_formatter())
        setattr(file_handler, _HANDLER_MARKER, True)
        logger.addHandler(file_handler)
