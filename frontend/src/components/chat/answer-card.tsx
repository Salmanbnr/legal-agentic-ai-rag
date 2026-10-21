"use client";

import { useEffect, useId, useRef, useState } from "react";

import { HowFound } from "@/components/chat/how-found";
import { answerToPlainText } from "@/lib/answer-copy";
import { citationLabel, citationsForClaim } from "@/lib/citation";
import type { AnswerCitation, AnswerRecord, FinalStatus } from "@/lib/types";

const BANNERS: Partial<
  Record<FinalStatus, { title: string; body: string; className: string }>
> = {
  insufficient_evidence: {
    title: "Not in the files",
    body:
      "The case file does not contain enough evidence to answer this question. Nothing has been guessed.",
    className: "border-amber-200 bg-amber-50 text-amber-950",
  },
  conflicting_evidence: {
    title: "The documents disagree",
    body:
      "The file supports more than one reading. Both are shown below, and neither is presented as settled.",
    className: "border-gold/40 bg-gold/10 text-ink",
  },
};

function SectionHeading({
  id,
  children,
}: {
  id: string;
  children: React.ReactNode;
}) {
  return (
    <h3
      id={id}
      className="text-[11px] font-semibold uppercase tracking-wide text-muted"
    >
      {children}
    </h3>
  );
}

function CitationChip({
  citation,
  onOpen,
}: {
  citation: AnswerCitation;
  onOpen: (citation: AnswerCitation) => void;
}) {
  return (
    <button
      type="button"
      onClick={() => onOpen(citation)}
      aria-label={`View the evidence from ${citationLabel(citation)}`}
      className="inline-flex max-w-full items-center gap-1 rounded-full border border-line bg-background px-2.5 py-1 text-[11px] font-medium text-foreground transition hover:border-accent/40 hover:bg-accent-soft"
    >
      <span className="truncate">{citationLabel(citation)}</span>
    </button>
  );
}

function ClaimList({
  claims,
  citations,
  onOpenCitation,
}: {
  claims: AnswerRecord["answer"]["claims"];
  citations: AnswerCitation[];
  onOpenCitation: (citation: AnswerCitation) => void;
}) {
  return (
    <ul className="mt-2 space-y-2.5">
      {claims.map((claim, index) => {
        const chips = citationsForClaim(claim.evidence_ids, citations);
        return (
          <li key={`${claim.text}-${index}`} className="space-y-1.5">
            <p className="text-[15px] leading-7">{claim.text}</p>
            {chips.length > 0 ? (
              <div className="flex flex-wrap gap-1.5">
                {chips.map((citation) => (
                  <CitationChip
                    key={`${citation.evidence_id}-${index}`}
                    citation={citation}
                    onOpen={onOpenCitation}
                  />
                ))}
              </div>
            ) : null}
          </li>
        );
      })}
    </ul>
  );
}

/** Return the short answer text, falling back to the verified claims. */
function shortAnswer(answer: AnswerRecord["answer"]): string {
  const text = answer.answer_text?.trim();
  if (text) {
    return text;
  }
  return answer.claims.map((claim) => claim.text).join("\n") || "";
}

/**
 * One answer, rendered as the sections a lawyer reads in order: the short
 * answer, the evidence behind it, any conflict, the limits, and the next step.
 *
 * Direct document statements and inferences are styled differently, so a
 * reading of the file is never mistaken for a statement in it.
 */
export function AnswerCard({
  record,
  onOpenCitation,
}: {
  record: AnswerRecord;
  onOpenCitation: (citation: AnswerCitation) => void;
}) {
  const headingId = useId();
  const [copyState, setCopyState] = useState<"idle" | "copied" | "error">(
    "idle",
  );
  const copyTimer = useRef<number | null>(null);

  useEffect(
    () => () => {
      if (copyTimer.current !== null) {
        window.clearTimeout(copyTimer.current);
      }
    },
    [],
  );

  const answer = record.answer;
  const banner = BANNERS[answer.final_status];
  const direct = answer.claims.filter((claim) => claim.type === "direct");
  const inference = answer.claims.filter((claim) => claim.type === "inference");
  const referenced = new Set(answer.claims.flatMap((claim) => claim.evidence_ids));
  const unreferenced = answer.evidence.filter(
    (citation) => !referenced.has(citation.evidence_id),
  );
  const summary = shortAnswer(answer);

  async function copyAnswer() {
    try {
      if (!navigator.clipboard?.writeText) {
        throw new Error("clipboard unavailable");
      }
      await navigator.clipboard.writeText(answerToPlainText(record));
      setCopyState("copied");
    } catch {
      setCopyState("error");
    }
    if (copyTimer.current !== null) {
      window.clearTimeout(copyTimer.current);
    }
    copyTimer.current = window.setTimeout(() => setCopyState("idle"), 2500);
  }

  return (
    <article
      aria-labelledby={headingId}
      data-status={answer.final_status}
      className="rounded-2xl border border-line bg-surface px-5 py-5"
    >
      <header className="flex flex-wrap items-center justify-between gap-2">
        <span className="rounded-full border border-accent/20 bg-accent-soft px-2.5 py-1 text-[11px] font-semibold text-accent">
          {record.mode === "agentic" ? "Agentic" : "Standard"}
        </span>
        <div className="flex items-center gap-2">
          <span aria-live="polite" className="text-[11px] text-muted">
            {copyState === "copied"
              ? "Copied with citations"
              : copyState === "error"
                ? "Could not copy. Please select the text instead."
                : ""}
          </span>
          <button
            type="button"
            onClick={() => void copyAnswer()}
            className="rounded-full border border-line px-3 py-1 text-[12px] font-medium text-muted transition hover:text-foreground"
          >
            Copy answer
          </button>
        </div>
      </header>

      <div className="mt-4 space-y-5">
        {banner ? (
          <div
            role="note"
            className={`rounded-xl border px-4 py-3 text-sm leading-6 ${banner.className}`}
          >
            <p className="font-semibold">{banner.title}</p>
            <p className="mt-0.5">{banner.body}</p>
          </div>
        ) : null}

        <section aria-labelledby={`${headingId}-short`}>
          <SectionHeading id={`${headingId}-short`}>Short answer</SectionHeading>
          {summary ? (
            <p className="mt-1.5 whitespace-pre-line text-[15px] leading-7">
              {summary}
            </p>
          ) : (
            <p className="mt-1.5 text-[15px] leading-7 text-muted">
              No answer was produced from this case file.
            </p>
          )}
        </section>

        <section aria-labelledby={`${headingId}-evidence`}>
          <SectionHeading id={`${headingId}-evidence`}>
            Supporting evidence
          </SectionHeading>

          {direct.length > 0 ? (
            <div className="mt-2 border-l-2 border-accent/60 pl-3">
              <p className="text-[11px] font-semibold uppercase tracking-wide text-accent">
                Stated in the documents
              </p>
              <ClaimList
                claims={direct}
                citations={answer.evidence}
                onOpenCitation={onOpenCitation}
              />
            </div>
          ) : null}

          {inference.length > 0 ? (
            <div className="mt-3 border-l-2 border-dashed border-gold/70 pl-3">
              <p className="text-[11px] font-semibold uppercase tracking-wide text-gold">
                Inference — the assistant&rsquo;s reading, not a statement in
                the file
              </p>
              <div className="italic">
                <ClaimList
                  claims={inference}
                  citations={answer.evidence}
                  onOpenCitation={onOpenCitation}
                />
              </div>
            </div>
          ) : null}

          {answer.claims.length === 0 && answer.evidence.length === 0 ? (
            <p className="mt-1.5 text-[15px] leading-7 text-muted">
              No page in the case file supports an answer to this question.
            </p>
          ) : null}

          {unreferenced.length > 0 ? (
            <div className="mt-3">
              <p className="text-[11px] font-semibold uppercase tracking-wide text-muted">
                Other pages used
              </p>
              <div className="mt-1.5 flex flex-wrap gap-1.5">
                {unreferenced.map((citation) => (
                  <CitationChip
                    key={citation.evidence_id}
                    citation={citation}
                    onOpen={onOpenCitation}
                  />
                ))}
              </div>
            </div>
          ) : null}
        </section>

        {answer.conflicts.length > 0 ? (
          <section aria-labelledby={`${headingId}-conflicts`}>
            <SectionHeading id={`${headingId}-conflicts`}>
              Conflicts
            </SectionHeading>
            <ul className="mt-1.5 list-disc space-y-1 pl-5 text-[15px] leading-7">
              {answer.conflicts.map((conflict, index) => (
                <li key={`${conflict}-${index}`}>{conflict}</li>
              ))}
            </ul>
          </section>
        ) : null}

        <section aria-labelledby={`${headingId}-limits`}>
          <SectionHeading id={`${headingId}-limits`}>
            Limitations
          </SectionHeading>
          {answer.limitations.length > 0 ? (
            <ul className="mt-1.5 list-disc space-y-1 pl-5 text-[14px] leading-6 text-muted">
              {answer.limitations.map((limitation, index) => (
                <li key={`${limitation}-${index}`}>{limitation}</li>
              ))}
            </ul>
          ) : (
            <p className="mt-1.5 text-[14px] leading-6 text-muted">
              None noted.
            </p>
          )}
        </section>

        <section aria-labelledby={`${headingId}-follow-up`}>
          <SectionHeading id={`${headingId}-follow-up`}>
            Suggested follow-up
          </SectionHeading>
          {answer.follow_up_questions.length > 0 ? (
            <ul className="mt-1.5 space-y-1.5">
              {answer.follow_up_questions.map((question, index) => (
                <li
                  key={`${question}-${index}`}
                  className="rounded-lg border border-line bg-background/60 px-3 py-2 text-[14px] leading-6"
                >
                  {question}
                </li>
              ))}
            </ul>
          ) : (
            <p className="mt-1.5 text-[14px] leading-6 text-muted">
              No follow-up suggested.
            </p>
          )}
        </section>

        {record.mode === "agentic" && record.trace ? (
          <HowFound trace={record.trace} />
        ) : null}
      </div>
    </article>
  );
}
