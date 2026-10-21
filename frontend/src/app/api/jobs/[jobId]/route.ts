import { callBackend, withSession } from "@/lib/api";
import type { JobSummary } from "@/lib/types";

type Context = { params: Promise<{ jobId: string }> };

/** Return one processing job. */
export async function GET(request: Request, context: Context) {
  return withSession(request, async (api) => {
    const { jobId } = await context.params;
    const job = await callBackend<JobSummary>(
      `/api/v1/jobs/${encodeURIComponent(jobId)}`,
      api,
    );
    return job;
  });
}
