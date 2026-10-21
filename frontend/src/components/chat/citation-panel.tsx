"use client";

import { useCallback, useEffect, useId, useRef, useState } from "react";

import { extractionLabel, pageLabel } from "@/lib/citation";
import type {
  AnswerCitation,
  EvidenceDetail,
  SignedUrlResponse,
} from "@/lib/types";

const PANEL_TIMEOUT_MS = 15_000;

const SAFE_LOAD_ERROR =
  "We could not load the exact text just now. Please try again.";
const SAFE_OPEN_ERROR =
  "We could not open the original file. Please try again in a moment.";

async function readMessage(response: Response, fallback: string) {
  const body = (await response.json().catch(() => null)) as {
    message?: string;
  } | null;
  return body?.message?.trim() || fallback;
}

/**
 * The evidence side panel of one citation.
 *
 * The text is read from the stored chunk when the panel opens, not from the
 * answer, so it is always the record itself. The original is opened through a
 * fresh 60-second signed link requested on demand.
 */
export function CitationPanel({
  caseId,
  citation,
  onClose,
}: {
  caseId: string;
  citation: AnswerCitation | null;
  onClose: () => void;
}) {
  if (!citation) {
    return null;
  }
  // The key resets the panel state when a different citation is opened, so the
  // body can load on mount instead of clearing state inside an effect.
  return (
    <PanelBody
      key={citation.evidence_id}
      caseId={caseId}
      citation={citation}
      onClose={onClose}
    />
  );
}

function PanelBody({
  caseId,
  citation,
  onClose,
}: {
  caseId: string;
  citation: AnswerCitation;
  onClose: () => void;
}) {
  const titleId = useId();
  const closeRef = useRef<HTMLButtonElement>(null);
  const [detail, setDetail] = useState<EvidenceDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [opening, setOpening] = useState(false);
  const [openError, setOpenError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);

  const evidenceId = citation.evidence_id;

  useEffect(() => {
    let active = true;
    const controller = new AbortController();
    const timer = window.setTimeout(() => controller.abort(), PANEL_TIMEOUT_MS);

    fetch(
      `/api/cases/${encodeURIComponent(caseId)}/evidence/${encodeURIComponent(evidenceId)}`,
      { cache: "no-store", signal: controller.signal },
    )
      .then(async (response) => {
        if (!active) {
          return;
        }
        if (!response.ok) {
          setError(await readMessage(response, SAFE_LOAD_ERROR));
          return;
        }
        setDetail((await response.json()) as EvidenceDetail);
      })
      .catch(() => {
        if (active) {
          setError(SAFE_LOAD_ERROR);
        }
      })
      .finally(() => {
        window.clearTimeout(timer);
        if (active) {
          setLoading(false);
        }
      });

    return () => {
      active = false;
      controller.abort();
      window.clearTimeout(timer);
    };
  }, [caseId, evidenceId, attempt]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onClose();
      }
    };
    document.addEventListener("keydown", onKeyDown);
    const timer = window.setTimeout(() => closeRef.current?.focus(), 0);
    return () => {
      document.removeEventListener("keydown", onKeyDown);
      window.clearTimeout(timer);
    };
  }, [onClose]);

  const openOriginal = useCallback(async () => {
    setOpening(true);
    setOpenError(null);
    try {
      const response = await fetch(
        `/api/documents/${encodeURIComponent(citation.document_id)}/signed-url`,
        { cache: "no-store", signal: AbortSignal.timeout(PANEL_TIMEOUT_MS) },
      );
      if (!response.ok) {
        setOpenError(await readMessage(response, SAFE_OPEN_ERROR));
        return;
      }
      const signed = (await response.json()) as SignedUrlResponse;
      const page = detail?.page_start ?? citation.page_start;
      const mime = signed.mime ?? detail?.document_mime ?? "";
      const target =
        mime.includes("pdf") && page ? `${signed.url}#page=${page}` : signed.url;
      window.open(target, "_blank", "noopener,noreferrer");
    } catch {
      setOpenError(SAFE_OPEN_ERROR);
    } finally {
      setOpening(false);
    }
  }, [citation, detail]);

  const shown = detail;
  const page = shown ? pageLabel({ ...citation, ...shown }) : pageLabel(citation);

  return (
    <div className="fixed inset-0 z-50">
      <button
        type="button"
        aria-label="Close the evidence panel"
        onClick={onClose}
        className="absolute inset-0 cursor-default bg-ink/40 backdrop-blur-sm"
      />
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        className="absolute inset-y-0 right-0 flex w-full max-w-lg flex-col border-l border-line bg-surface shadow-[-30px_0_80px_-60px_rgba(19,32,25,0.7)]"
      >
        <header className="flex items-start justify-between gap-4 border-b border-line px-5 py-4">
          <div>
            <p className="text-[11px] font-semibold uppercase tracking-wide text-muted">
              Evidence
            </p>
            <h2 id={titleId} className="mt-1 font-serif text-2xl leading-tight">
              {shown?.document_title ?? (citation.title || "Cited document")}
            </h2>
            <p className="mt-1 text-[13px] text-muted">
              {page || "Page not recorded"}
            </p>
          </div>
          <button
            ref={closeRef}
            type="button"
            onClick={onClose}
            className="rounded-full border border-line px-3 py-1 text-[13px] text-muted transition hover:text-foreground"
          >
            Close
          </button>
        </header>

        <div className="flex-1 space-y-4 overflow-y-auto px-5 py-5">
          {loading ? (
            <p role="status" className="text-sm text-muted">
              Loading the exact text…
            </p>
          ) : null}

          {error ? (
            <div
              role="alert"
              className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800"
            >
              <p>{error}</p>
              <button
                type="button"
                onClick={() => {
                  setLoading(true);
                  setError(null);
                  setAttempt((previous) => previous + 1);
                }}
                className="mt-2 rounded-full border border-red-300 px-3 py-1 text-[13px] font-semibold"
              >
                Try again
              </button>
            </div>
          ) : null}

          {shown ? (
            <>
              <dl className="grid grid-cols-[minmax(0,9rem)_1fr] gap-y-2 text-[13px]">
                <dt className="text-muted">Pages</dt>
                <dd>{pageLabel({ ...citation, ...shown }) || "Not recorded"}</dd>
                <dt className="text-muted">Extraction</dt>
                <dd>{extractionLabel(shown.extraction_method)}</dd>
                {shown.section_path.length > 0 ? (
                  <>
                    <dt className="text-muted">Section</dt>
                    <dd>{shown.section_path.join(" › ")}</dd>
                  </>
                ) : null}
              </dl>

              {shown.quality_warnings.length > 0 ? (
                <div
                  role="note"
                  className="rounded-xl border border-amber-200 bg-amber-50/80 px-4 py-3 text-[13px] leading-6 text-amber-950"
                >
                  <p className="font-semibold">Check this page against the original.</p>
                  <ul className="mt-1.5 list-disc space-y-1 pl-5">
                    {shown.quality_warnings.map((warning) => (
                      <li key={warning}>{warning}</li>
                    ))}
                  </ul>
                </div>
              ) : null}

              <figure className="rounded-xl border border-line bg-background/60 px-4 py-4">
                <blockquote className="whitespace-pre-line font-serif text-[15px] leading-7">
                  {shown.text}
                </blockquote>
                <figcaption className="mt-3 text-[11px] text-muted">
                  Stored text from {shown.document_title}
                  {page ? `, ${page}` : ""}
                </figcaption>
              </figure>
            </>
          ) : null}

          {openError ? (
            <p
              role="alert"
              className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800"
            >
              {openError}
            </p>
          ) : null}
        </div>

        <footer className="flex flex-wrap items-center justify-between gap-3 border-t border-line px-5 py-4">
          <p className="text-[11px] text-muted">
            The link expires after one minute.
          </p>
          <button
            type="button"
            onClick={() => void openOriginal()}
            disabled={opening}
            className="rounded-full bg-accent px-4 py-2 text-sm font-semibold text-surface transition hover:bg-accent/90 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {opening ? "Opening…" : "Open original"}
          </button>
        </footer>
      </div>
    </div>
  );
}
