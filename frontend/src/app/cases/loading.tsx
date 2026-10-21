/** The loading state of the case list. */
export default function CasesLoading() {
  return (
    <div className="flex min-h-screen flex-col">
      <div className="h-[86px] border-b border-line/80 bg-background/75" />
      <main className="mx-auto w-full max-w-6xl flex-1 px-6 py-10">
        <div
          role="status"
          aria-live="polite"
          className="space-y-3"
        >
          <p className="text-sm text-muted">Loading your cases…</p>
          <div className="h-10 w-64 animate-pulse rounded-xl bg-line/60" />
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {[0, 1, 2].map((item) => (
              <div
                key={item}
                className="h-36 animate-pulse rounded-2xl border border-line bg-surface/70"
              />
            ))}
          </div>
        </div>
      </main>
    </div>
  );
}
