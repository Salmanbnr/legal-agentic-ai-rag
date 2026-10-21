import type { AnswerTrace } from "@/lib/types";

/**
 * The collapsible workflow panel of an agentic answer.
 *
 * It shows how the answer was looked for: the plan, the queries that were
 * tried, the retry count, and the nodes that ran. It never shows evidence text,
 * prompts, or completions.
 */
export function HowFound({ trace }: { trace: AnswerTrace }) {
  return (
    <details className="rounded-xl border border-line bg-background/70 text-[13px]">
      <summary className="flex cursor-pointer list-none items-center justify-between gap-3 px-4 py-3 font-medium">
        <span>How this was found</span>
        <span className="text-[11px] font-normal text-muted">
          {trace.steps} steps · {trace.retries_used}{" "}
          {trace.retries_used === 1 ? "retry" : "retries"}
        </span>
      </summary>

      <div className="space-y-4 border-t border-line px-4 py-4">
        <div>
          <h4 className="text-[11px] font-semibold uppercase tracking-wide text-muted">
            Plan
          </h4>
          {trace.plan.length === 0 ? (
            <p className="mt-1 text-muted">No plan was recorded.</p>
          ) : (
            <ol className="mt-1.5 list-decimal space-y-1 pl-5">
              {trace.plan.map((step, index) => (
                <li key={`${step.text}-${index}`}>
                  {step.text}
                  {step.purpose ? (
                    <span className="text-muted"> — {step.purpose}</span>
                  ) : null}
                </li>
              ))}
            </ol>
          )}
        </div>

        <div>
          <h4 className="text-[11px] font-semibold uppercase tracking-wide text-muted">
            Queries tried
          </h4>
          {trace.queries.length === 0 ? (
            <p className="mt-1 text-muted">No queries were recorded.</p>
          ) : (
            <ul className="mt-1.5 space-y-1">
              {trace.queries.map((query, index) => (
                <li
                  key={`${query}-${index}`}
                  className="rounded-lg border border-line bg-surface px-2.5 py-1.5"
                >
                  {query}
                </li>
              ))}
            </ul>
          )}
        </div>

        <div>
          <h4 className="text-[11px] font-semibold uppercase tracking-wide text-muted">
            Nodes visited
          </h4>
          {trace.node_path.length === 0 ? (
            <p className="mt-1 text-muted">No nodes were recorded.</p>
          ) : (
            <ol className="mt-1.5 flex flex-wrap gap-1.5">
              {trace.node_path.map((node, index) => (
                <li
                  key={`${node}-${index}`}
                  className="rounded-full border border-line bg-surface px-2.5 py-1 text-[11px]"
                >
                  {node}
                </li>
              ))}
            </ol>
          )}
        </div>

        <p className="text-[11px] text-muted">
          Retries: {trace.retries_used} · LLM calls: {trace.llm_calls} · Workflow:{" "}
          {trace.graph_version}
        </p>
      </div>
    </details>
  );
}
