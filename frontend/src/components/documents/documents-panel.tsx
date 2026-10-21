"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { DocumentTable } from "@/components/documents/document-table";
import { UploadDropzone } from "@/components/documents/upload-dropzone";
import { friendlyMessage } from "@/lib/messages";
import type { DocumentSummary, JobSummary } from "@/lib/types";

const POLL_INTERVAL_MS = 3000;

type DocumentsPanelProps = {
  caseId: string;
  initialDocuments: DocumentSummary[];
};

/** Return true while any document is still being worked on. */
export function isProcessing(documents: DocumentSummary[]): boolean {
  return documents.some((document) => {
    if (document.deletion_requested) {
      return false;
    }
    return (
      document.processing_status !== "ready" &&
      document.processing_status !== "failed" &&
      document.processing_status !== "needs_review"
    );
  });
}

/**
 * The documents tab.
 *
 * While anything is processing the list refreshes every three seconds, so the
 * stage and the quality badges stay current without a manual refresh.
 */
export function DocumentsPanel({
  caseId,
  initialDocuments,
}: DocumentsPanelProps) {
  const [documents, setDocuments] = useState(initialDocuments);
  const [seed, setSeed] = useState(initialDocuments);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [activeJobs, setActiveJobs] = useState<string[]>([]);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Adopt a new server list without an effect, so the panel never renders a
  // stale list for a frame.
  if (seed !== initialDocuments) {
    setSeed(initialDocuments);
    setDocuments(initialDocuments);
  }

  const refresh = useCallback(
    async (showSpinner = false) => {
      if (showSpinner) {
        setLoading(true);
      }
      try {
        const response = await fetch(
          `/api/cases/${encodeURIComponent(caseId)}/documents`,
          { cache: "no-store", signal: AbortSignal.timeout(15_000) },
        );
        if (!response.ok) {
          const body = (await response.json().catch(() => null)) as {
            message?: string;
          } | null;
          setError(
            body?.message ?? "We could not load the documents just now.",
          );
          return;
        }
        const next = (await response.json()) as DocumentSummary[];
        setDocuments(next);
        setError(null);
      } catch {
        setError("We could not reach the service. Please try again.");
      } finally {
        setLoading(false);
      }
    },
    [caseId],
  );

  useEffect(() => {
    if (!isProcessing(documents) && activeJobs.length === 0) {
      return;
    }
    timerRef.current = setTimeout(() => {
      void refresh();
    }, POLL_INTERVAL_MS);
    return () => {
      if (timerRef.current) {
        clearTimeout(timerRef.current);
        timerRef.current = null;
      }
    };
  }, [documents, activeJobs, refresh]);

  const registerJob = useCallback((jobId: string | null) => {
    if (!jobId) {
      return;
    }
    setActiveJobs((current) =>
      current.includes(jobId) ? current : [...current, jobId],
    );
  }, []);

  useEffect(() => {
    if (activeJobs.length === 0) {
      return;
    }
    let cancelled = false;
    void (async () => {
      for (const jobId of activeJobs) {
        try {
          const response = await fetch(
            `/api/jobs/${encodeURIComponent(jobId)}`,
            { cache: "no-store", signal: AbortSignal.timeout(15_000) },
          );
          if (!response.ok) {
            continue;
          }
          const job = (await response.json()) as JobSummary;
          if (cancelled) {
            return;
          }
          if (job.status === "succeeded" || job.status === "failed") {
            setActiveJobs((current) =>
              current.filter((value) => value !== jobId),
            );
            if (job.status === "failed") {
              setError(
                friendlyMessage(
                  job.error_code ?? undefined,
                  "We could not finish reading that document. Please try again.",
                ),
              );
            }
            await refresh();
          }
        } catch {
          // The next poll tries again.
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [activeJobs, refresh]);

  return (
    <section aria-labelledby="documents-heading" className="space-y-6">
      <div>
        <h2 id="documents-heading" className="font-serif text-2xl tracking-tight">
          Documents
        </h2>
        <p className="mt-1.5 text-sm text-muted">
          The files that make up this case. Each one is read once, then indexed
          so you can ask questions about it.
        </p>
      </div>

      <UploadDropzone
        caseId={caseId}
        onUploaded={(jobId) => {
          registerJob(jobId);
          void refresh();
        }}
      />

      {error ? (
        <p
          role="alert"
          className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800"
        >
          {error}
        </p>
      ) : null}

      {loading ? (
        <p className="text-sm text-muted" role="status">
          Refreshing the list…
        </p>
      ) : null}

      <DocumentTable
        documents={documents}
        onChanged={() => void refresh(true)}
      />
    </section>
  );
}
