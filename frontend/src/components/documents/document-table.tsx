"use client";

import { useState } from "react";

import { Dialog } from "@/components/dialog";
import { ProcessingBadge, QualityBadge } from "@/components/documents/badges";
import { QualityWarning } from "@/components/documents/quality-warning";
import { formatBytes, formatDate, formatFileType } from "@/lib/format";
import type { DocumentSummary } from "@/lib/types";

type DocumentTableProps = {
  documents: DocumentSummary[];
  onChanged?: () => void;
};

/** The list of documents in a case, with its status and quality badges. */
export function DocumentTable({ documents, onChanged }: DocumentTableProps) {
  const [pendingDelete, setPendingDelete] = useState<DocumentSummary | null>(
    null,
  );
  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  async function confirmDelete() {
    if (!pendingDelete) {
      return;
    }
    setDeleting(true);
    setDeleteError(null);
    try {
      const response = await fetch(
        `/api/documents/${encodeURIComponent(pendingDelete.id)}`,
        { method: "DELETE" },
      );
      if (!response.ok) {
        const body = (await response.json().catch(() => null)) as {
          message?: string;
        } | null;
        setDeleteError(
          body?.message ?? "We could not delete that document. Please try again.",
        );
        setDeleting(false);
        return;
      }
      setPendingDelete(null);
      setDeleting(false);
      onChanged?.();
    } catch {
      setDeleteError("We could not reach the service. Please try again.");
      setDeleting(false);
    }
  }

  if (documents.length === 0) {
    return (
      <div className="rounded-2xl border border-dashed border-line bg-surface/60 px-6 py-12 text-center">
        <p className="font-serif text-xl">No documents yet</p>
        <p className="mx-auto mt-2 max-w-sm text-sm leading-6 text-muted">
          Upload the case record above. The original file is kept exactly as
          you sent it, and every answer points back to the page it came from.
        </p>
      </div>
    );
  }

  return (
    <>
      <div className="overflow-hidden rounded-2xl border border-line bg-surface">
        <table className="w-full border-collapse text-left text-sm">
          <caption className="sr-only">
            Documents in this case, with their processing and quality status
          </caption>
          <thead>
            <tr className="border-b border-line bg-background/60 text-[11px] uppercase tracking-[0.12em] text-muted">
              <th scope="col" className="px-4 py-3 font-semibold">
                File
              </th>
              <th scope="col" className="hidden px-4 py-3 font-semibold md:table-cell">
                Type
              </th>
              <th scope="col" className="hidden px-4 py-3 font-semibold sm:table-cell">
                Pages
              </th>
              <th scope="col" className="hidden px-4 py-3 font-semibold sm:table-cell">
                Size
              </th>
              <th scope="col" className="px-4 py-3 font-semibold">
                Status
              </th>
              <th scope="col" className="px-4 py-3 font-semibold">
                Quality
              </th>
              <th scope="col" className="px-4 py-3 text-right font-semibold">
                <span className="sr-only">Actions</span>
              </th>
            </tr>
          </thead>
          <tbody>
            {documents.map((document) => (
              <tr
                key={document.id}
                className="border-b border-line/70 align-top last:border-0"
              >
                <th scope="row" className="px-4 py-3 font-normal">
                  <p className="font-medium text-foreground">
                    {document.original_filename}
                  </p>
                  <p className="mt-0.5 text-xs text-muted">
                    Added {formatDate(document.created_at)}
                    {document.duplicate_of ? " · a copy of an earlier file" : ""}
                  </p>
                  <QualityWarning document={document} />
                </th>
                <td className="hidden px-4 py-3 text-muted md:table-cell">
                  {formatFileType(document.detected_mime)}
                </td>
                <td className="hidden px-4 py-3 text-muted sm:table-cell">
                  {document.page_count ?? "—"}
                </td>
                <td className="hidden px-4 py-3 text-muted sm:table-cell">
                  {formatBytes(document.size_bytes)}
                </td>
                <td className="px-4 py-3">
                  <ProcessingBadge status={document.processing_status} />
                </td>
                <td className="px-4 py-3">
                  <QualityBadge status={document.quality_status} />
                </td>
                <td className="px-4 py-3 text-right">
                  <button
                    type="button"
                    onClick={() => {
                      setDeleteError(null);
                      setPendingDelete(document);
                    }}
                    className="rounded-full border border-line px-3 py-1.5 text-xs font-medium text-muted transition hover:border-red-300 hover:text-red-700"
                  >
                    Delete
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <Dialog
        open={pendingDelete !== null}
        title="Delete this document?"
        description="The document, its pages, and its search index entries are removed. The original file is deleted from storage as well. This cannot be undone."
        onClose={() => {
          if (!deleting) {
            setPendingDelete(null);
          }
        }}
        footer={
          <>
            <button
              type="button"
              onClick={() => setPendingDelete(null)}
              disabled={deleting}
              className="rounded-full border border-line px-4 py-2 text-sm font-medium text-muted transition hover:text-foreground"
            >
              Keep it
            </button>
            <button
              type="button"
              onClick={confirmDelete}
              disabled={deleting}
              className="rounded-full bg-red-700 px-4 py-2 text-sm font-medium text-white transition hover:bg-red-800 disabled:opacity-60"
            >
              {deleting ? "Deleting…" : "Delete document"}
            </button>
          </>
        }
      >
        <p className="text-sm text-foreground">
          {pendingDelete?.original_filename}
        </p>
        {deleteError ? (
          <p
            role="alert"
            className="mt-3 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800"
          >
            {deleteError}
          </p>
        ) : null}
      </Dialog>
    </>
  );
}
