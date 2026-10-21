import { callBackend, withSession } from "@/lib/api";
import { BACKEND_UPLOAD_TIMEOUT_MS } from "@/lib/backend";
import type { DocumentSummary, UploadResponse } from "@/lib/types";

type Context = { params: Promise<{ caseId: string }> };

/** List the documents in a case. */
export async function GET(request: Request, context: Context) {
  return withSession(request, async (api) => {
    const { caseId } = await context.params;
    const documents = await callBackend<DocumentSummary[]>(
      `/api/v1/cases/${encodeURIComponent(caseId)}/documents?limit=100&offset=0`,
      api,
    );
    return documents;
  });
}

/** Upload one or more files into a case. */
export async function POST(request: Request, context: Context) {
  return withSession(request, async (api) => {
    const { caseId } = await context.params;
    const form = await request.formData();
    const files = form.getAll("file").filter((value): value is File => value instanceof File);

    if (files.length === 0) {
      return Response.json(
        {
          error: "NO_FILE",
          message: "Please choose a file to upload.",
          request_id: api.requestId,
        },
        { status: 422, headers: { "X-Request-ID": api.requestId } },
      );
    }

    const results: UploadResponse[] = [];
    for (const file of files) {
      const body = new FormData();
      body.append("file", file, file.name);
      const result = await callBackend<UploadResponse>(
        `/api/v1/cases/${encodeURIComponent(caseId)}/documents`,
        api,
        {
          method: "POST",
          body,
          timeoutMs: BACKEND_UPLOAD_TIMEOUT_MS,
        },
      );
      results.push(result);
    }

    return results;
  });
}
