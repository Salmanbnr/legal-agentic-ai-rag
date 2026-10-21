import Link from "next/link";

import { AppHeader } from "@/components/app-header";
import { CreateCaseDialog } from "@/components/create-case-dialog";
import { backendFetch } from "@/lib/backend";
import { formatDate } from "@/lib/format";
import { getSession } from "@/lib/session";
import type { CaseSummary } from "@/lib/types";

export const metadata = {
  title: "Cases · Legal Evidence Workspace",
};

/**
 * The list of cases.
 *
 * The list is read on the server with the signed-in user's token, so the
 * browser never sees the backend address.
 */
export default async function CasesPage() {
  const session = await getSession();
  const email = session?.user.email ?? null;

  let cases: CaseSummary[] = [];
  let error: string | null = null;

  if (session) {
    try {
      cases = await backendFetch<CaseSummary[]>("/api/v1/cases", {
        token: session.token,
      });
    } catch {
      error =
        "We could not load your cases just now. Please refresh the page and try again.";
    }
  }

  return (
    <div className="flex min-h-screen flex-col">
      <AppHeader email={email} />

      <main className="mx-auto w-full max-w-6xl flex-1 px-6 py-10">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <h1 className="font-serif text-4xl tracking-tight">Your cases</h1>
            <p className="mt-2 max-w-xl text-sm leading-6 text-muted">
              Each case keeps its own documents and its own answers. Only you
              can see them.
            </p>
          </div>
          <CreateCaseDialog />
        </div>

        {error ? (
          <p
            role="alert"
            className="mt-8 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800"
          >
            {error}
          </p>
        ) : null}

        {!error && cases.length === 0 ? (
          <div className="mt-10 rounded-2xl border border-dashed border-line bg-surface/60 px-6 py-14 text-center">
            <p className="font-serif text-2xl">No cases yet</p>
            <p className="mx-auto mt-2 max-w-md text-sm leading-6 text-muted">
              Create a case for the matter you are working on, then upload the
              documents you want to ask about.
            </p>
          </div>
        ) : null}

        {cases.length > 0 ? (
          <ul className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {cases.map((item) => (
              <li key={item.id}>
                <Link
                  href={`/cases/${item.id}`}
                  className="block h-full rounded-2xl border border-line bg-surface p-5 transition hover:border-accent/40"
                >
                  <h2 className="font-serif text-xl leading-snug">
                    {item.title}
                  </h2>
                  {item.description ? (
                    <p className="mt-2 line-clamp-2 text-sm leading-6 text-muted">
                      {item.description}
                    </p>
                  ) : null}
                  <p className="mt-4 text-xs text-muted">
                    Opened {formatDate(item.created_at)}
                    {item.status === "archived" ? " · archived" : ""}
                  </p>
                </Link>
              </li>
            ))}
          </ul>
        ) : null}
      </main>
    </div>
  );
}
