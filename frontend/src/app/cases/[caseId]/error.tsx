"use client";

/** The error state of one case. */
export default function CaseError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <div className="rounded-2xl border border-red-200 bg-red-50/80 px-6 py-10 text-center">
      <h2 className="font-serif text-2xl">We could not open this case</h2>
      <p className="mx-auto mt-2 max-w-md text-sm leading-6 text-red-900">
        The case may have been removed, or the service may be briefly
        unavailable. Please try again.
      </p>
      <button
        type="button"
        onClick={reset}
        className="mt-5 rounded-full bg-red-700 px-5 py-2.5 text-sm font-medium text-white transition hover:bg-red-800"
      >
        Try again
      </button>
      {error.digest ? (
        <p className="mt-4 font-mono text-[10px] uppercase tracking-[0.14em] text-red-900/60">
          Reference {error.digest}
        </p>
      ) : null}
    </div>
  );
}
