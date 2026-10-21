import { DocumentsPanel } from "@/components/documents/documents-panel";
import { backendFetch } from "@/lib/backend";
import { getSession } from "@/lib/session";
import type { DocumentSummary } from "@/lib/types";

/** The documents tab of one case. */
export default async function CaseDocumentsPage({
  params,
}: PageProps<"/cases/[caseId]">) {
  const { caseId } = await params;
  const session = await getSession();

  let documents: DocumentSummary[] = [];
  let error: string | null = null;

  if (session) {
    try {
      documents = await backendFetch<DocumentSummary[]>(
        `/api/v1/cases/${encodeURIComponent(caseId)}/documents?limit=100&offset=0`,
        { token: session.token },
      );
    } catch {
      error =
        "We could not load the documents just now. Please refresh the page and try again.";
    }
  }

  if (error) {
    return (
      <p
        role="alert"
        className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800"
      >
        {error}
      </p>
    );
  }

  return <DocumentsPanel caseId={caseId} initialDocuments={documents} />;
}
