import Link from "next/link";

/** The page shown when a case or document cannot be found. */
export default function NotFound() {
  return (
    <main className="mx-auto flex min-h-screen w-full max-w-xl flex-col items-center justify-center px-6 text-center">
      <p className="font-mono text-[11px] uppercase tracking-[0.2em] text-muted">
        Not found
      </p>
      <h1 className="mt-4 font-serif text-4xl tracking-tight">
        We could not find that
      </h1>
      <p className="mt-3 max-w-md text-sm leading-6 text-muted">
        The page may have been removed, or the link may be out of date. Your
        case room is still where you left it.
      </p>
      <Link
        href="/cases"
        className="mt-7 rounded-full bg-ink px-5 py-2.5 text-sm font-medium text-surface transition hover:bg-accent"
      >
        Back to your cases
      </Link>
    </main>
  );
}
