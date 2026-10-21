import { callBackend, withSession } from "@/lib/api";
import { BACKEND_ANSWER_TIMEOUT_MS } from "@/lib/backend";

type Context = { params: Promise<{ caseId: string }> };

/** Ask a question about a case. */
export async function POST(request: Request, context: Context) {
  return withSession(request, async (api) => {
    const { caseId } = await context.params;
    const payload = (await request.json().catch(() => null)) as {
      question?: unknown;
      mode?: unknown;
    } | null;

    const question =
      typeof payload?.question === "string" ? payload.question.trim() : "";
    if (!question) {
      return Response.json(
        {
          error: "EMPTY_QUESTION",
          message: "Please type a question.",
          request_id: api.requestId,
        },
        { status: 422, headers: { "X-Request-ID": api.requestId } },
      );
    }

    const mode = payload?.mode === "agentic" ? "agentic" : "baseline";
    const answer = await callBackend<unknown>(
      `/api/v1/cases/${encodeURIComponent(caseId)}/ask`,
      api,
      {
        method: "POST",
        body: JSON.stringify({ question, mode }),
        timeoutMs: BACKEND_ANSWER_TIMEOUT_MS,
      },
    );
    return answer;
  });
}
