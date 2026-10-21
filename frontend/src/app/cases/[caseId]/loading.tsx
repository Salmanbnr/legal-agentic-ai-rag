/** The loading state of one case. */
export default function CaseLoading() {
  return (
    <div role="status" aria-live="polite" className="space-y-4">
      <p className="text-sm text-muted">Loading the case…</p>
      <div className="h-10 w-72 animate-pulse rounded-xl bg-line/60" />
      <div className="h-40 w-full animate-pulse rounded-2xl border border-line bg-surface/70" />
    </div>
  );
}
