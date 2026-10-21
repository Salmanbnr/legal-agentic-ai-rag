import { callBackend, withSession } from "@/lib/api";
import type { DocumentSummary } from "@/lib/types";

type Context = { params: Promise<{ documentId: string }> };

/** Return one document. */
export async function GET(request: Request, context: Context) {
  return withSession(request, async (api) => {
    const { documentId } = await context.params;
    const document = await callBackend<DocumentSummary>(
      `/api/v1/documents/${encodeURIComponent(documentId)}`,
      api,
    );
    return document;
  });
}

/** Ask the backend to delete a document. */
export async function DELETE(request: Request, context: Context) {
  return withSession(request, async (api) => {
    const { documentId } = await context.params;
    await callBackend<void>(
      `/api/v1/documents/${encodeURIComponent(documentId)}`,
      api,
      { method: "DELETE" },
    );
    return { deleted: true };
  });
}
