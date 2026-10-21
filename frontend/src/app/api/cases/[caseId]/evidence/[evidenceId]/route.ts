import { callBackend, withSession } from "@/lib/api";
import type { EvidenceDetail } from "@/lib/types";

type Context = { params: Promise<{ caseId: string; evidenceId: string }> };

/** Return the exact stored evidence behind one citation. */
export async function GET(request: Request, context: Context) {
  return withSession(request, async (api) => {
    const { caseId, evidenceId } = await context.params;
    return callBackend<EvidenceDetail>(
      `/api/v1/cases/${encodeURIComponent(caseId)}/evidence/${encodeURIComponent(evidenceId)}`,
      api,
    );
  });
}
