import { callBackend, withSession } from "@/lib/api";
import type { CaseSummary } from "@/lib/types";

/** List the cases owned by the signed-in user. */
export async function GET(request: Request) {
  return withSession(request, async (context) => {
    const cases = await callBackend<CaseSummary[]>("/api/v1/cases", context);
    return cases;
  });
}

/** Create a case. */
export async function POST(request: Request) {
  return withSession(request, async (context) => {
    const payload = (await request.json().catch(() => null)) as {
      title?: unknown;
      description?: unknown;
    } | null;

    const title = typeof payload?.title === "string" ? payload.title.trim() : "";
    if (!title) {
      return Response.json(
        {
          error: "INVALID_TITLE",
          message: "Please give the case a name.",
          request_id: context.requestId,
        },
        { status: 422, headers: { "X-Request-ID": context.requestId } },
      );
    }

    const description =
      typeof payload?.description === "string"
        ? payload.description.trim()
        : "";

    const created = await callBackend<CaseSummary>("/api/v1/cases", context, {
      method: "POST",
      body: JSON.stringify({ title, description }),
    });
    return created;
  });
}
