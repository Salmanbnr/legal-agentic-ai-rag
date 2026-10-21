import { formatPageNumbers } from "@/lib/format";
import type { DocumentSummary } from "@/lib/types";

/**
 * The warning panel for a document that needs attention.
 *
 * It names the pages that were hard to read, so a lawyer knows where to look
 * before relying on the text.
 */
export function QualityWarning({ document }: { document: DocumentSummary }) {
  const report = document.quality_report;
  const lowConfidence = report?.low_confidence_pages ?? [];
  const emptyPages = report?.empty_pages ?? [];
  const parserErrors = report?.parser_errors ?? [];

  const needsReview = document.quality_status === "needs_review";
  const hasWarnings =
    lowConfidence.length > 0 ||
    emptyPages.length > 0 ||
    parserErrors.length > 0 ||
    document.quality_status === "ready_with_warnings" ||
    document.quality_status === "password_required" ||
    needsReview;

  if (!hasWarnings) {
    return null;
  }

  return (
    <div
      role="note"
      className="mt-3 rounded-xl border border-amber-200 bg-amber-50/80 px-4 py-3 text-[13px] leading-6 text-amber-950"
    >
      <p className="font-semibold">
        {needsReview
          ? "This document needs a second look."
          : "Some pages in this document were hard to read."}
      </p>

      {lowConfidence.length > 0 ? (
        <p className="mt-1.5">
          The text on page {formatPageNumbers(lowConfidence)} was read with low
          confidence, so a name, a date, or an amount may be wrong. Check those
          pages against the original before relying on them.
        </p>
      ) : null}

      {emptyPages.length > 0 ? (
        <p className="mt-1.5">
          No text could be read from page {formatPageNumbers(emptyPages)}. The
          page may be an image or a scan.
        </p>
      ) : null}

      {parserErrors.length > 0 ? (
        <p className="mt-1.5">
          Some pages could not be processed. Open the original to see what they
          contain.
        </p>
      ) : null}

      {document.quality_status === "password_required" ? (
        <p className="mt-1.5">
          This file is password protected, so nothing could be read from it.
        </p>
      ) : null}
    </div>
  );
}
