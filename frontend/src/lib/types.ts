/** Shared types that mirror the backend API responses. */

export type QualityStatus =
  | "pending"
  | "ready"
  | "ready_with_warnings"
  | "needs_review"
  | "failed"
  | "password_required";

export type ProcessingStatus =
  | "uploaded"
  | "validating"
  | "extracting"
  | "ocr"
  | "chunking"
  | "embedding"
  | "indexing"
  | "ready"
  | "needs_review"
  | "failed";

export type JobStatus = "queued" | "running" | "succeeded" | "failed";

export interface CaseSummary {
  id: string;
  title: string;
  description: string | null;
  status: "active" | "archived";
  created_at: string;
  updated_at: string;
}

export interface QualityReportSummary {
  pages_total: number | null;
  pages_extracted: number | null;
  pages_empty: number | null;
  empty_pages: number[];
  low_confidence_pages: number[];
  repeated_text_ratio: number | null;
  parser_errors: { page_number?: number; error_code?: string }[];
}

export interface DocumentSummary {
  id: string;
  case_id: string;
  original_filename: string;
  detected_mime: string | null;
  size_bytes: number | null;
  page_count: number | null;
  document_type: string | null;
  duplicate_of: string | null;
  quality_status: QualityStatus;
  processing_status: ProcessingStatus;
  deletion_requested: boolean;
  quality_report: QualityReportSummary | null;
  created_at: string;
  updated_at: string;
}

export interface JobSummary {
  id: string;
  case_id: string;
  document_id: string;
  job_type: "ingest" | "reindex" | "delete";
  status: JobStatus;
  attempts: number;
  error_code: string | null;
  created_at: string;
}

export interface UploadResponse {
  document_id: string;
  job_id: string | null;
  quality_status: QualityStatus;
  duplicate_of: string | null;
  error_code: string | null;
}

export interface ApiErrorBody {
  error: string;
  message: string;
  code?: string;
  request_id?: string;
}

export type FinalStatus =
  | "answered"
  | "partial"
  | "insufficient_evidence"
  | "conflicting_evidence";

export type AnswerMode = "baseline" | "agentic";

export type ClaimType = "direct" | "inference";

/** One citation as returned with an answer. */
export interface AnswerCitation {
  evidence_id: string;
  case_id: string;
  document_id: string;
  document_version_id: string | null;
  title: string;
  page_start: number | null;
  page_end: number | null;
  section_path: string[];
  quote_or_span: string;
  extraction_method: string | null;
  quality_status: QualityStatus;
  ocr_confidence: number | null;
  source_anchor: string | null;
  retrieval_method: string;
  retrieval_rank: number;
  quality_flags: unknown[];
  retrieval_score: number | null;
}

export interface AnswerClaimView {
  text: string;
  evidence_ids: string[];
  type: ClaimType;
  confidence: number | null;
}

export interface AnswerSections {
  answer_text: string;
  claims: AnswerClaimView[];
  conflicts: string[];
  limitations: string[];
  follow_up_questions: string[];
  final_status: FinalStatus;
  reason_codes: string[];
  evidence: AnswerCitation[];
}

export interface TracePlanStep {
  text: string;
  purpose: string;
  status: string;
}

/** The safe workflow summary shown in the "how this was found" panel. */
export interface AnswerTrace {
  graph_version: string;
  node_path: string[];
  plan: TracePlanStep[];
  queries: string[];
  steps: number;
  llm_calls: number;
  tokens_used: number;
  retries_used: number;
  duration_ms: number;
  final_status: FinalStatus;
  reason_codes: string[];
}

/** One stored answer, as returned by the ask and history endpoints. */
export interface AnswerRecord {
  answer_id: string;
  case_id: string;
  question: string;
  answer: AnswerSections;
  mode: AnswerMode;
  trace: AnswerTrace | null;
  created_at?: string;
}

export interface AnswersPage {
  items: AnswerRecord[];
  limit: number;
  offset: number;
  has_more: boolean;
}

/** The exact stored evidence behind one citation. */
export interface EvidenceDetail {
  evidence_id: string;
  case_id: string;
  document_id: string;
  document_title: string;
  document_mime: string | null;
  page_start: number | null;
  page_end: number | null;
  section_path: string[];
  text: string;
  extraction_method: string | null;
  ocr_confidence: number | null;
  quality_status: QualityStatus;
  quality_flags: unknown[];
  quality_warnings: string[];
  source_anchor: string | null;
}

export interface SignedUrlResponse {
  document_id: string;
  filename: string;
  mime: string | null;
  url: string;
  expires_in_seconds: number;
}
