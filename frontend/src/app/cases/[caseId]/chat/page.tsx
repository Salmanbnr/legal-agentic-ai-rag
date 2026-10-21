import { ChatPanel } from "@/components/chat/chat-panel";
import { backendFetch } from "@/lib/backend";
import { getSession } from "@/lib/session";
import type { AnswersPage, AnswerRecord } from "@/lib/types";

const INITIAL_PAGE_SIZE = 20;

/** The chat tab of one case. */
export default async function CaseChatPage({
  params,
}: PageProps<"/cases/[caseId]/chat">) {
  const { caseId } = await params;
  const session = await getSession();

  let answers: AnswerRecord[] = [];
  let hasMore = false;
  let error: string | null = null;

  if (session) {
    try {
      const page = await backendFetch<AnswersPage>(
        `/api/v1/cases/${encodeURIComponent(caseId)}/answers?limit=${INITIAL_PAGE_SIZE}&offset=0`,
        { token: session.token },
      );
      // The endpoint lists newest first; the conversation reads oldest first.
      answers = [...page.items].reverse();
      hasMore = page.has_more;
    } catch {
      error =
        "We could not load the earlier answers just now. You can still ask a new question.";
    }
  }

  return (
    <section aria-label="Ask this case" className="space-y-6">
      {error ? (
        <p
          role="alert"
          className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-950"
        >
          {error}
        </p>
      ) : null}

      <ChatPanel
        caseId={caseId}
        initialAnswers={answers}
        initialHasMore={hasMore}
      />
    </section>
  );
}
