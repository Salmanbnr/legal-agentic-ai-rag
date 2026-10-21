import { callBackend, withSession } from "@/lib/api";
import type { SignedUrlResponse } from "@/lib/types";

type Context = { params: Promise<{ documentId: string }> };

/** Mint a short-lived link to the original document. */
export async function GET(request: Request, context: Context) {
  return withSession(request, async (api) => {
    const { documentId } = await context.params;
    return callBackend<SignedUrlResponse>(
      `/api/v1/documents/${encodeURIComponent(documentId)}/signed-url`,
      api,
    );
  });
}
