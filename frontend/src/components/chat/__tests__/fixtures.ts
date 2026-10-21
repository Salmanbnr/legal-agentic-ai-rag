import type {
  AnswerCitation,
  AnswerRecord,
  AnswerSections,
  AnswerTrace,
} from "@/lib/types";

export function makeCitation(
  overrides: Partial<AnswerCitation> = {},
): AnswerCitation {
  return {
    evidence_id: "ev-1",
    case_id: "case-1",
    document_id: "doc-1",
    document_version_id: null,
    title: "Lease agreement.pdf",
    page_start: 7,
    page_end: 7,
    section_path: ["Clause 12"],
    quote_or_span: "The monthly rent is GBP 1,250.",
    extraction_method: "ocr",
    quality_status: "ready_with_warnings",
    ocr_confidence: 58,
    source_anchor: "doc:1|p:7|c:0",
    retrieval_method: "hybrid",
    retrieval_rank: 1,
    quality_flags: [],
    retrieval_score: 0.04,
    ...overrides,
  };
}

export function makeTrace(overrides: Partial<AnswerTrace> = {}): AnswerTrace {
  return {
    graph_version: "agentic-rag-3",
    node_path: ["authorize", "route", "build_plan", "retrieve", "final_gate"],
    plan: [
      {
        text: "What rent does clause 12 set?",
        purpose: "Find the rent clause.",
        status: "retrieved",
      },
    ],
    queries: ["What rent does clause 12 set?"],
    steps: 5,
    llm_calls: 3,
    tokens_used: 900,
    retries_used: 1,
    duration_ms: 1200,
    final_status: "answered",
    reason_codes: [],
    ...overrides,
  };
}

export function makeRecord(
  answerOverrides: Partial<AnswerSections> = {},
  recordOverrides: Partial<Omit<AnswerRecord, "answer">> = {},
): AnswerRecord {
  const citation = makeCitation();
  return {
    answer_id: "answer-1",
    case_id: "case-1",
    question: "What is the monthly rent?",
    mode: "baseline",
    trace: null,
    created_at: "2026-01-01T10:00:00+00:00",
    ...recordOverrides,
    answer: {
      answer_text: "The rent is GBP 1,250 per month.",
      claims: [
        {
          text: "Clause 12 sets the rent at GBP 1,250 per month.",
          evidence_ids: [citation.evidence_id],
          type: "direct",
          confidence: null,
        },
      ],
      conflicts: [],
      limitations: [],
      follow_up_questions: [],
      final_status: "answered",
      reason_codes: [],
      evidence: [citation],
      ...answerOverrides,
    },
  };
}
