import { callBackend, withSession } from "@/lib/api";
import type { CaseSummary } from "@/lib/types";

type Context = { params: Promise<{ caseId: string }> };

/** Return one case. */
export async function GET(request: Request, context: Context) {
  return withSession(request, async (api) => {
    const { caseId } = await context.params;
    const found = await callBackend<CaseSummary>(
      `/api/v1/cases/${encodeURIComponent(caseId)}`,
      api,
    );
    return found;
  });
}
