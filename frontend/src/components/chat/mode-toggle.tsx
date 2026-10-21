"use client";

import type { AnswerMode } from "@/lib/types";

const OPTIONS: { value: AnswerMode; label: string; hint: string }[] = [
  {
    value: "baseline",
    label: "Standard",
    hint: "One search over the case file. Fast.",
  },
  {
    value: "agentic",
    label: "Agentic",
    hint: "Plans the question, searches more than once, then checks every claim.",
  },
];

/** Choose between a single search and the multi-step agent workflow. */
export function ModeToggle({
  mode,
  onChange,
  disabled,
}: {
  mode: AnswerMode;
  onChange: (mode: AnswerMode) => void;
  disabled?: boolean;
}) {
  const active = OPTIONS.find((option) => option.value === mode) ?? OPTIONS[0];
  return (
    <div className="flex flex-wrap items-center gap-3">
      <div
        role="group"
        aria-label="Answering mode"
        className="inline-flex rounded-full border border-line bg-background p-1"
      >
        {OPTIONS.map((option) => {
          const selected = option.value === mode;
          return (
            <button
              key={option.value}
              type="button"
              aria-pressed={selected}
              disabled={disabled}
              onClick={() => onChange(option.value)}
              className={`rounded-full px-3.5 py-1.5 text-[13px] font-medium transition disabled:cursor-not-allowed disabled:opacity-60 ${
                selected
                  ? "bg-accent text-surface"
                  : "text-muted hover:text-foreground"
              }`}
            >
              {option.label}
            </button>
          );
        })}
      </div>
      <p className="text-[13px] text-muted">{active.hint}</p>
    </div>
  );
}
