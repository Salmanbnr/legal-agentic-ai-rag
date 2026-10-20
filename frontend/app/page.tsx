const principles = [
  {
    title: "Private by construction",
    body: "Supabase Auth, private Storage, and backend-enforced case ownership. No teams, no shared tenancy.",
  },
  {
    title: "Source-bound answers",
    body: "Retrieved excerpts drive every response. When the record is thin, the workspace abstains instead of guessing.",
  },
  {
    title: "Originals stay untouched",
    body: "Uploaded files are immutable. Every extraction and index record stays traceable to its source page.",
  },
];

const foundation = [
  "Next.js App Router shell with Tailwind",
  "FastAPI /health, request ids, and safe JSON errors",
  "Environment validation plus a pytest suite",
];

function CheckIcon() {
  return (
    <svg
      aria-hidden="true"
      viewBox="0 0 20 20"
      fill="none"
      className="size-4 shrink-0"
    >
      <path
        d="m4 10.5 3.5 3.5L16 5.5"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export default function Home() {
  return (
    <div className="flex min-h-screen flex-col">
      <header className="border-b border-line/80 bg-background/75 backdrop-blur-md">
        <div className="mx-auto flex w-full max-w-6xl items-center justify-between gap-4 px-6 py-5">
          <div className="flex items-center gap-3">
            <div className="flex size-10 items-center justify-center rounded-xl bg-ink font-serif text-xl text-surface shadow-sm">
              §
            </div>
            <div>
              <p className="font-serif text-lg leading-none tracking-tight">
                Legal Evidence Workspace
              </p>
              <p className="mt-1.5 text-[10px] font-medium uppercase tracking-[0.22em] text-muted">
                Private case room
              </p>
            </div>
          </div>
          <span className="hidden items-center gap-2 rounded-full border border-line bg-surface/80 px-3.5 py-1.5 text-xs font-medium text-muted sm:inline-flex">
            <span className="size-1.5 rounded-full bg-accent" />
            Single user · Local preview
          </span>
        </div>
      </header>

      <main className="flex-1">
        <section className="mx-auto grid w-full max-w-6xl gap-14 px-6 py-16 lg:grid-cols-[1.04fr_0.96fr] lg:items-center lg:gap-16 lg:py-24">
          <div>
            <span className="inline-flex items-center gap-2 rounded-full border border-accent/20 bg-accent-soft px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.16em] text-accent">
              Built for one lawyer
            </span>
            <h1 className="mt-6 max-w-xl font-serif text-5xl leading-[1.03] tracking-[-0.025em] text-balance sm:text-6xl">
              Every answer, traced to the page.
            </h1>
            <p className="mt-6 max-w-lg text-[15px] leading-7 text-muted sm:text-base">
              Upload the case record. Ask in plain language. Review answers with
              citations that point back to the source document and page — never
              to an invented reference.
            </p>

            <div className="mt-8 flex flex-wrap items-center gap-3">
              <a
                href="#foundation"
                className="inline-flex items-center gap-2 rounded-full bg-ink px-5 py-3 text-sm font-medium text-surface transition-colors hover:bg-accent"
              >
                View build status
                <svg
                  aria-hidden="true"
                  viewBox="0 0 16 16"
                  fill="none"
                  className="size-4"
                >
                  <path
                    d="M3 8h10M9 4l4 4-4 4"
                    stroke="currentColor"
                    strokeWidth="1.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              </a>
              <a
                href="#principles"
                className="inline-flex items-center rounded-full border border-line bg-surface/70 px-5 py-3 text-sm font-medium text-ink transition-colors hover:border-accent/40 hover:bg-surface"
              >
                Read the guardrails
              </a>
            </div>

            <dl className="mt-10 grid max-w-lg grid-cols-3 gap-4 border-t border-line pt-6">
              <div>
                <dt className="text-[10px] font-semibold uppercase tracking-[0.16em] text-muted">
                  Access
                </dt>
                <dd className="mt-1.5 text-sm font-medium">Backend enforced</dd>
              </div>
              <div>
                <dt className="text-[10px] font-semibold uppercase tracking-[0.16em] text-muted">
                  Storage
                </dt>
                <dd className="mt-1.5 text-sm font-medium">Private by default</dd>
              </div>
              <div>
                <dt className="text-[10px] font-semibold uppercase tracking-[0.16em] text-muted">
                  Answers
                </dt>
                <dd className="mt-1.5 text-sm font-medium">Page cited</dd>
              </div>
            </dl>
          </div>

          <div className="relative">
            <div
              aria-hidden="true"
              className="absolute -inset-3 -rotate-1 rounded-[28px] border border-gold/20 bg-surface/40"
            />
            <div className="relative overflow-hidden rounded-2xl border border-line bg-surface shadow-[0_30px_90px_-45px_rgba(19,32,25,0.5)]">
              <div className="flex items-center justify-between border-b border-line px-4 py-3">
                <div className="flex items-center gap-1.5" aria-hidden="true">
                  <span className="size-2 rounded-full bg-line" />
                  <span className="size-2 rounded-full bg-line" />
                  <span className="size-2 rounded-full bg-line" />
                </div>
                <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-muted">
                  Case room
                </span>
                <span className="inline-flex items-center gap-1.5 rounded-full bg-accent-soft px-2 py-1 text-[9px] font-semibold uppercase tracking-[0.12em] text-accent">
                  <span className="size-1 rounded-full bg-accent" />
                  Private
                </span>
              </div>

              <div className="grid grid-cols-[7.5rem_1fr]">
                <div className="border-r border-line/80 p-3">
                  {["Case files", "Ask the record", "Citations"].map(
                    (item, index) => (
                      <div
                        key={item}
                        className={`mb-1 rounded-lg px-2.5 py-2 text-[11px] font-medium ${
                          index === 0
                            ? "bg-ink text-surface"
                            : "text-muted"
                        }`}
                      >
                        {item}
                      </div>
                    ),
                  )}
                  <div className="mt-4 border-t border-line/70 pt-3">
                    <div className="px-2 text-[9px] font-semibold uppercase tracking-[0.16em] text-muted/70">
                      Case
                    </div>
                    <div className="mt-2 space-y-2 px-2" aria-hidden="true">
                      <div className="h-1.5 w-16 rounded-full bg-line" />
                      <div className="h-1.5 w-12 rounded-full bg-line/70" />
                    </div>
                  </div>
                </div>

                <div className="p-5">
                  <div className="flex items-start justify-between gap-4">
                    <div>
                      <p className="font-serif text-xl leading-tight">
                        No case files yet
                      </p>
                      <p className="mt-1.5 text-xs leading-5 text-muted">
                        Uploads and page-linked evidence will appear here.
                      </p>
                    </div>
                    <div
                      aria-hidden="true"
                      className="flex size-10 shrink-0 items-center justify-center rounded-xl border border-line bg-background"
                    >
                      <svg
                        viewBox="0 0 24 24"
                        fill="none"
                        className="size-5 text-accent"
                      >
                        <path
                          d="M14 3v5h5M7 3h8l4 4v14H7V3Z"
                          stroke="currentColor"
                          strokeWidth="1.4"
                          strokeLinejoin="round"
                        />
                        <path
                          d="M10 12h6M10 16h6"
                          stroke="currentColor"
                          strokeWidth="1.4"
                          strokeLinecap="round"
                        />
                      </svg>
                    </div>
                  </div>

                  <div className="mt-5 rounded-xl border border-dashed border-line bg-background/70 px-4 py-7 text-center">
                    <div
                      aria-hidden="true"
                      className="mx-auto flex size-9 items-center justify-center rounded-full bg-accent-soft text-accent"
                    >
                      <svg
                        viewBox="0 0 20 20"
                        fill="none"
                        className="size-4"
                      >
                        <path
                          d="M10 4v9m0 0 3.5-3.5M10 13 6.5 9.5M4 16h12"
                          stroke="currentColor"
                          strokeWidth="1.5"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        />
                      </svg>
                    </div>
                    <p className="mt-3 text-xs font-medium text-ink">
                      Drop case files to begin
                    </p>
                    <p className="mt-1 text-[10px] text-muted">
                      PDF · Original preserved
                    </p>
                  </div>

                  <div className="mt-4 flex items-center gap-2 rounded-xl border border-line bg-background/80 px-3 py-2.5">
                    <span className="text-[11px] text-muted">
                      Ask a question about this case…
                    </span>
                    <span
                      aria-hidden="true"
                      className="ml-auto flex size-7 items-center justify-center rounded-full bg-ink text-surface"
                    >
                      <svg
                        viewBox="0 0 16 16"
                        fill="none"
                        className="size-3.5"
                      >
                        <path
                          d="M3 8h10M9 4l4 4-4 4"
                          stroke="currentColor"
                          strokeWidth="1.5"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        />
                      </svg>
                    </span>
                  </div>
                </div>
              </div>

              <div className="border-t border-line bg-background/60 px-4 py-2.5 text-center text-[10px] text-muted">
                Original files stay untouched · Citations resolve to stored pages
              </div>
            </div>
          </div>
        </section>

        <section
          id="principles"
          className="mx-auto w-full max-w-6xl scroll-mt-20 px-6 pb-16"
        >
          <div className="max-w-xl">
            <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-accent">
              Built for the record
            </p>
            <h2 className="mt-3 font-serif text-3xl tracking-tight text-balance sm:text-4xl">
              Three rules the workspace does not bend.
            </h2>
          </div>

          <div className="mt-8 grid gap-4 md:grid-cols-3">
            {principles.map((principle, index) => (
              <article
                key={principle.title}
                className="rounded-2xl border border-line bg-surface/80 p-6"
              >
                <span className="font-mono text-[10px] font-medium text-gold">
                  0{index + 1}
                </span>
                <h3 className="mt-4 font-serif text-xl tracking-tight">
                  {principle.title}
                </h3>
                <p className="mt-2.5 text-[13px] leading-6 text-muted">
                  {principle.body}
                </p>
              </article>
            ))}
          </div>
        </section>

        <section
          id="foundation"
          className="mx-auto w-full max-w-6xl scroll-mt-20 px-6 pb-20"
        >
          <div className="grid gap-8 rounded-2xl border border-line bg-ink p-7 text-surface sm:p-9 md:grid-cols-[1fr_1fr] md:items-center">
            <div>
              <p className="text-[10px] font-semibold uppercase tracking-[0.2em] text-surface/50">
                Scaffold status
              </p>
              <h2 className="mt-3 font-serif text-3xl tracking-tight text-balance sm:text-4xl">
                The foundation is in place.
              </h2>
              <p className="mt-4 max-w-md text-sm leading-6 text-surface/65">
                Upload, extraction, indexing, and grounded retrieval come next.
                The security and evidence rules are already wired into the
                project.
              </p>
            </div>
            <ul className="space-y-3">
              {foundation.map((item) => (
                <li
                  key={item}
                  className="flex items-center gap-3 rounded-xl border border-surface/10 bg-surface/5 px-4 py-3 text-sm text-surface/85"
                >
                  <span className="flex size-6 items-center justify-center rounded-full bg-accent-soft text-accent">
                    <CheckIcon />
                  </span>
                  {item}
                </li>
              ))}
            </ul>
          </div>
        </section>
      </main>

      <footer className="border-t border-line bg-surface/50">
        <div className="mx-auto flex w-full max-w-6xl flex-col gap-3 px-6 py-6 text-xs text-muted sm:flex-row sm:items-center sm:justify-between">
          <p>Legal Evidence Workspace · Single lawyer · Private case room</p>
          <p className="font-mono text-[10px] uppercase tracking-[0.14em]">
            Next.js · FastAPI · Supabase · Pinecone
          </p>
        </div>
      </footer>
    </div>
  );
}
