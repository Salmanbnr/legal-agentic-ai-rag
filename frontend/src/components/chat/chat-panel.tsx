"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { AnswerCard } from "@/components/chat/answer-card";
import { CitationPanel } from "@/components/chat/citation-panel";
import { Composer } from "@/components/chat/composer";
import { messageForStatus } from "@/lib/messages";
import type {
  AnswerCitation,
  AnswerMode,
  AnswerRecord,
  AnswersPage,
} from "@/lib/types";

/** The browser gives up a little after the server deadline, so the server wins. */
const CLIENT_TIMEOUT_MS = 125_000;
const SLOW_AFTER_MS = 12_000;
const HISTORY_PAGE_SIZE = 20;

const SAFE_REACH_ERROR =
  "We could not reach the service. Please check your connection and try again.";
const SLOW_MESSAGE =
  "This is taking longer than expected. Agentic answers can take up to two minutes; you can keep waiting or try again.";

type ChatError = { message: string; retryable: boolean };

async function readErrorMessage(response: Response): Promise<string> {
  const body = (await response.json().catch(() => null)) as {
    message?: string;
  } | null;
  const message = body?.message?.trim();
  if (message) {
    return message;
  }
  return messageForStatus(response.status);
}

/**
 * The chat tab of one case.
 *
 * History is loaded from the stored answers, and each new answer is appended as
 * soon as it arrives. The component keeps the only fetch logic, so a slow
 * request always ends in a message and a retry button instead of a spinner that
 * never stops.
 */
export function ChatPanel({
  caseId,
  initialAnswers,
  initialHasMore,
}: {
  caseId: string;
  initialAnswers: AnswerRecord[];
  initialHasMore: boolean;
}) {
  const [records, setRecords] = useState<AnswerRecord[]>(initialAnswers);
  const [mode, setMode] = useState<AnswerMode>("baseline");
  const [question, setQuestion] = useState("");
  const [pending, setPending] = useState(false);
  const [pendingQuestion, setPendingQuestion] = useState<string | null>(null);
  const [slow, setSlow] = useState(false);
  const [error, setError] = useState<ChatError | null>(null);
  const [lastAttempt, setLastAttempt] = useState<{
    question: string;
    mode: AnswerMode;
  } | null>(null);
  const [selectedCitation, setSelectedCitation] =
    useState<AnswerCitation | null>(null);
  const [hasMore, setHasMore] = useState(initialHasMore);
  const [loadingOlder, setLoadingOlder] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);

  const ask = useCallback(
    async (text: string, selectedMode: AnswerMode) => {
      const trimmed = text.trim();
      if (!trimmed) {
        return;
      }
      setPending(true);
      setPendingQuestion(trimmed);
      setSlow(false);
      setError(null);
      setLastAttempt({ question: trimmed, mode: selectedMode });

      try {
        const response = await fetch(`/api/cases/${caseId}/ask`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          cache: "no-store",
          body: JSON.stringify({ question: trimmed, mode: selectedMode }),
          signal: AbortSignal.timeout(CLIENT_TIMEOUT_MS),
        });

        if (!response.ok) {
          if (response.status === 504) {
            setError({
              message: `This is taking too long. ${messageForStatus(504)}`,
              retryable: true,
            });
            return;
          }
          setError({
            message: await readErrorMessage(response),
            retryable: response.status !== 401 && response.status !== 404,
          });
          return;
        }

        const record = (await response.json()) as AnswerRecord;
        setRecords((previous) => [...previous, record]);
        setQuestion("");
      } catch (caught) {
        const timedOut =
          caught instanceof Error &&
          (caught.name === "TimeoutError" || caught.name === "AbortError");
        setError({
          message: timedOut
            ? "This is taking longer than expected, so we stopped waiting. Please try again."
            : SAFE_REACH_ERROR,
          retryable: true,
        });
      } finally {
        setPending(false);
        setPendingQuestion(null);
        setSlow(false);
      }
    },
    [caseId],
  );

  useEffect(() => {
    if (!pending) {
      return;
    }
    const timer = window.setTimeout(() => setSlow(true), SLOW_AFTER_MS);
    return () => window.clearTimeout(timer);
  }, [pending]);

  useEffect(() => {
    endRef.current?.scrollIntoView?.({ behavior: "smooth", block: "end" });
  }, [records.length, pending]);

  const loadOlder = useCallback(async () => {
    setLoadingOlder(true);
    setError(null);
    try {
      const response = await fetch(
        `/api/cases/${caseId}/answers?limit=${HISTORY_PAGE_SIZE}&offset=${records.length}`,
        { cache: "no-store", signal: AbortSignal.timeout(15_000) },
      );
      if (!response.ok) {
        setError({
          message: await readErrorMessage(response),
          retryable: true,
        });
        return;
      }
      const page = (await response.json()) as AnswersPage;
      setHasMore(page.has_more);
      const older = [...page.items].reverse();
      setRecords((previous) => {
        const known = new Set(previous.map((record) => record.answer_id));
        return [...older.filter((record) => !known.has(record.answer_id)), ...previous];
      });
    } catch {
      setError({ message: SAFE_REACH_ERROR, retryable: true });
    } finally {
      setLoadingOlder(false);
    }
  }, [caseId, records.length]);

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h2 className="font-serif text-2xl tracking-tight">Ask this case</h2>
          <p className="mt-1.5 max-w-2xl text-sm leading-6 text-muted">
            Every answer comes with the pages it came from, and the workspace
            says so plainly when the record does not answer the question.
          </p>
        </div>
        {records.length > 0 ? (
          <p className="text-[11px] text-muted">
            {records.length}{" "}
            {records.length === 1 ? "answer" : "answers"} in this conversation
          </p>
        ) : null}
      </div>

      <div className="space-y-5">
        {hasMore ? (
          <div className="text-center">
            <button
              type="button"
              onClick={() => void loadOlder()}
              disabled={loadingOlder}
              className="rounded-full border border-line bg-surface px-3.5 py-1.5 text-[13px] font-medium text-muted transition hover:text-foreground disabled:cursor-not-allowed disabled:opacity-60"
            >
              {loadingOlder ? "Loading older answers…" : "Load older answers"}
            </button>
          </div>
        ) : null}

        {records.length === 0 && !pending ? (
          <div className="rounded-2xl border border-dashed border-line bg-surface/60 px-6 py-12 text-center">
            <p className="font-serif text-xl">No questions yet</p>
            <p className="mx-auto mt-2 max-w-md text-sm leading-6 text-muted">
              Ask about a date, a clause, an amount, or what the documents say.
              Answers are kept with this case, so you can come back to them.
            </p>
          </div>
        ) : null}

        {records.map((record) => (
          <div key={record.answer_id} className="space-y-2.5">
            <div className="flex flex-wrap items-baseline justify-between gap-2 px-1">
              <p className="max-w-3xl font-medium">
                <span className="text-muted">You asked: </span>
                {record.question}
              </p>
              {record.created_at ? (
                <time
                  dateTime={record.created_at}
                  className="text-[11px] text-muted"
                >
                  {new Date(record.created_at).toLocaleString("en-GB", {
                    day: "numeric",
                    month: "short",
                    hour: "2-digit",
                    minute: "2-digit",
                  })}
                </time>
              ) : null}
            </div>
            <AnswerCard record={record} onOpenCitation={setSelectedCitation} />
          </div>
        ))}

        {pendingQuestion ? (
          <div className="space-y-2.5">
            <p className="px-1 font-medium">
              <span className="text-muted">You asked: </span>
              {pendingQuestion}
            </p>
            <div className="rounded-2xl border border-line bg-surface px-5 py-5">
              <p role="status" className="flex items-center gap-2 text-sm text-muted">
                <span
                  aria-hidden="true"
                  className="inline-block h-3.5 w-3.5 animate-spin rounded-full border-2 border-line border-t-accent"
                />
                Searching the case…
              </p>
              {slow ? (
                <p className="mt-2 text-[13px] leading-6 text-muted">
                  {SLOW_MESSAGE}
                </p>
              ) : null}
            </div>
          </div>
        ) : null}

        <div ref={endRef} />
      </div>

      {error ? (
        <div
          role="alert"
          className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800"
        >
          <p>{error.message}</p>
          {error.retryable && lastAttempt ? (
            <button
              type="button"
              onClick={() => void ask(lastAttempt.question, lastAttempt.mode)}
              disabled={pending}
              className="rounded-full border border-red-300 px-3.5 py-1.5 text-[13px] font-semibold transition hover:bg-red-100 disabled:cursor-not-allowed disabled:opacity-60"
            >
              Try again
            </button>
          ) : null}
        </div>
      ) : null}

      <Composer
        value={question}
        mode={mode}
        pending={pending}
        onChange={setQuestion}
        onModeChange={setMode}
        onSubmit={() => void ask(question, mode)}
      />

      <CitationPanel
        caseId={caseId}
        citation={selectedCitation}
        onClose={() => setSelectedCitation(null)}
      />
    </div>
  );
}
