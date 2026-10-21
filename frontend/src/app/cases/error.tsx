"use client";

/** The error state of the case list. */
export default function CasesError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <main className="mx-auto flex min-h-screen w-full max-w-2xl flex-col items-center justify-center px-6 text-center">
      <h1 className="font-serif text-3xl tracking-tight">
        Something went wrong
      </h1>
      <p className="mt-3 text-sm leading-6 text-muted">
        We could not load your cases. This is usually temporary. Please try
        again, and tell us if it keeps happening.
      </p>
      <button
        type="button"
        onClick={reset}
        className="mt-6 rounded-full bg-ink px-5 py-2.5 text-sm font-medium text-surface transition hover:bg-accent"
      >
        Try again
      </button>
      {error.digest ? (
        <p className="mt-4 font-mono text-[10px] uppercase tracking-[0.14em] text-muted/70">
          Reference {error.digest}
        </p>
      ) : null}
    </main>
  );
}
