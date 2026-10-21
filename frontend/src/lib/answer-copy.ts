import { citationLabel } from "@/lib/citation";
import type { AnswerRecord, FinalStatus } from "@/lib/types";

const STATUS_NOTES: Record<FinalStatus, string | null> = {
  answered: null,
  partial:
    "Some of the question was answered, and some of it could not be supported by the file.",
  insufficient_evidence:
    "The file does not contain enough evidence to answer this question.",
  conflicting_evidence:
    "The documents disagree, so both readings are shown and neither is presented as settled.",
};

const MAX_CHARACTERS = 20000;

/**
 * Build the plain text version of an answer, citations included.
 *
 * The copy is meant to be pasted into a memo or an email, so every statement is
 * followed by the document and page it came from.
 */
export function answerToPlainText(record: AnswerRecord): string {
  const answer = record.answer;
  const lines: string[] = [];
  const add = (value: string) => lines.push(value);

  add(`Question: ${record.question}`);
  add("");
  add("Short answer:");
  add(answer.answer_text?.trim() || "No answer was produced.");
  add("");

  const direct = answer.claims.filter((claim) => claim.type === "direct");
  const inference = answer.claims.filter((claim) => claim.type === "inference");

  if (direct.length > 0) {
    add("Stated in the documents:");
    for (const claim of direct) {
      add(`- ${withCitations(claim.text, claim.evidence_ids, answer.evidence)}`);
    }
    add("");
  }

  if (inference.length > 0) {
    add("Inference (not a direct statement in the documents):");
    for (const claim of inference) {
      add(`- ${withCitations(claim.text, claim.evidence_ids, answer.evidence)}`);
    }
    add("");
  }

  if (answer.conflicts.length > 0) {
    add("Conflicts:");
    for (const conflict of answer.conflicts) {
      add(`- ${conflict}`);
    }
    add("");
  }

  if (answer.limitations.length > 0) {
    add("Limitations:");
    for (const limitation of answer.limitations) {
      add(`- ${limitation}`);
    }
    add("");
  }

  if (answer.follow_up_questions.length > 0) {
    add("Suggested follow-up:");
    for (const question of answer.follow_up_questions) {
      add(`- ${question}`);
    }
    add("");
  }

  const note = STATUS_NOTES[answer.final_status];
  if (note) {
    add(note);
    add("");
  }

  if (answer.evidence.length > 0) {
    add("Sources:");
    for (const citation of answer.evidence) {
      add(`- ${citationLabel(citation)}`);
    }
    add("");
  }

  add(`Mode: ${record.mode === "agentic" ? "Agentic" : "Standard"}`);
  return lines.join("\n").trim().slice(0, MAX_CHARACTERS);
}

function withCitations(
  text: string,
  evidenceIds: string[],
  citations: AnswerRecord["answer"]["evidence"],
): string {
  const byId = new Map(citations.map((item) => [item.evidence_id, item]));
  const labels: string[] = [];
  for (const id of evidenceIds) {
    const citation = byId.get(id);
    if (citation) {
      labels.push(citationLabel(citation));
    }
  }
  const suffix = labels.length > 0 ? ` [${labels.join("; ")}]` : "";
  return `${text.trim()}${suffix}`;
}
