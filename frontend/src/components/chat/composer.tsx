"use client";

import { ModeToggle } from "@/components/chat/mode-toggle";
import type { AnswerMode } from "@/lib/types";

/**
 * The question box.
 *
 * Enter sends and Shift+Enter starts a new line, which is the expectation for
 * a chat box. The shortcut is skipped while an input method is composing, so a
 * keyboard that uses Enter to confirm candidates still works.
 */
export function Composer({
  value,
  mode,
  pending,
  onChange,
  onModeChange,
  onSubmit,
}: {
  value: string;
  mode: AnswerMode;
  pending: boolean;
  onChange: (value: string) => void;
  onModeChange: (mode: AnswerMode) => void;
  onSubmit: () => void;
}) {
  const canSend = value.trim().length > 0 && !pending;

  return (
    <div className="rounded-2xl border border-line bg-surface p-4 shadow-[0_18px_50px_-40px_rgba(19,32,25,0.5)]">
      <label htmlFor="case-question" className="sr-only">
        Ask a question about this case
      </label>
      <textarea
        id="case-question"
        value={value}
        rows={3}
        disabled={pending}
        placeholder="Ask about a date, a clause, an amount, or what the documents say…"
        onChange={(event) => onChange(event.target.value)}
        onKeyDown={(event) => {
          if (
            event.key === "Enter" &&
            !event.shiftKey &&
            !event.nativeEvent.isComposing
          ) {
            event.preventDefault();
            if (canSend) {
              onSubmit();
            }
          }
        }}
        className="w-full resize-y bg-transparent text-[15px] leading-6 outline-none placeholder:text-muted/70 disabled:opacity-70"
      />

      <div className="mt-3 flex flex-wrap items-center justify-between gap-3 border-t border-line pt-3">
        <ModeToggle mode={mode} onChange={onModeChange} disabled={pending} />
        <div className="flex items-center gap-3">
          <p className="hidden text-[11px] text-muted sm:block">
            Enter to send · Shift+Enter for a new line
          </p>
          <button
            type="button"
            onClick={onSubmit}
            disabled={!canSend}
            className="rounded-full bg-accent px-5 py-2 text-sm font-semibold text-surface transition hover:bg-accent/90 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {pending ? "Searching…" : "Send"}
          </button>
        </div>
      </div>
    </div>
  );
}
