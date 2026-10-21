import { callBackend, withSession } from "@/lib/api";
import type { AnswersPage } from "@/lib/types";

type Context = { params: Promise<{ caseId: string }> };

/** Parse a bounded integer query value, or reject the request. */
function readInt(
  value: string | null,
  { fallback, min, max }: { fallback: number; min: number; max: number },
): number | null {
  if (value === null || value === "") {
    return fallback;
  }
  if (!/^\d+$/.test(value)) {
    return null;
  }
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed < min || parsed > max) {
    return null;
  }
  return parsed;
}

/** List the stored answers of a case, newest first. */
export async function GET(request: Request, context: Context) {
  return withSession(request, async (api) => {
    const { caseId } = await context.params;
    const url = new URL(request.url);
    const limit = readInt(url.searchParams.get("limit"), {
      fallback: 20,
      min: 1,
      max: 100,
    });
    const offset = readInt(url.searchParams.get("offset"), {
      fallback: 0,
      min: 0,
      max: 10_000,
    });

    if (limit === null || offset === null) {
      return Response.json(
        {
          error: "INVALID_PAGINATION",
          message: "That page of answers could not be read.",
          request_id: api.requestId,
        },
        { status: 422, headers: { "X-Request-ID": api.requestId } },
      );
    }

    return callBackend<AnswersPage>(
      `/api/v1/cases/${encodeURIComponent(caseId)}/answers?limit=${limit}&offset=${offset}`,
      api,
    );
  });
}
