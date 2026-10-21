import { notFound } from "next/navigation";

import { AppHeader } from "@/components/app-header";
import { CaseTabs } from "@/components/case-tabs";
import { backendFetch } from "@/lib/backend";
import { getSession } from "@/lib/session";
import type { CaseSummary } from "@/lib/types";

/**
 * The layout of one case.
 *
 * It shows the case name and the two areas of the case: the documents and
 * the chat.
 */
export default async function CaseLayout({
  children,
  params,
}: LayoutProps<"/cases/[caseId]">) {
  const { caseId } = await params;
  const session = await getSession();
  const email = session?.user.email ?? null;

  let found: CaseSummary | null = null;
  if (session) {
    try {
      found = await backendFetch<CaseSummary>(
        `/api/v1/cases/${encodeURIComponent(caseId)}`,
        { token: session.token },
      );
    } catch {
      found = null;
    }
  }

  if (!found) {
    notFound();
  }

  return (
    <div className="flex min-h-screen flex-col">
      <AppHeader email={email} />

      <main className="mx-auto w-full max-w-6xl flex-1 px-6 py-10">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <h1 className="font-serif text-4xl tracking-tight">{found.title}</h1>
            {found.description ? (
              <p className="mt-2 max-w-2xl text-sm leading-6 text-muted">
                {found.description}
              </p>
            ) : null}
          </div>
          <CaseTabs caseId={caseId} />
        </div>

        <div className="mt-8">{children}</div>
      </main>
    </div>
  );
}
