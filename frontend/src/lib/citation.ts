import type { AnswerCitation } from "@/lib/types";

/** Return "p. 7", "pp. 7–8", or an empty string when the page is unknown. */
export function pageLabel(citation: AnswerCitation): string {
  const start = citation.page_start;
  if (start === null || start === undefined) {
    return "";
  }
  const end = citation.page_end;
  if (end !== null && end !== undefined && end > start) {
    return `pp. ${start}–${end}`;
  }
  return `p. ${start}`;
}

/** Return "Contract.pdf, p. 7" for a citation chip. */
export function citationLabel(citation: AnswerCitation): string {
  const title = citation.title?.trim() || "Document";
  const page = pageLabel(citation);
  return page ? `${title}, ${page}` : title;
}

/** Return the plain language name of an extraction method. */
export function extractionLabel(method: string | null | undefined): string {
  switch (method) {
    case "native":
      return "Read from the digital text of the file";
    case "ocr":
      return "Read with OCR from a scan";
    case "mixed":
      return "Read from a mix of digital text and OCR";
    default:
      return "Extraction method was not recorded";
  }
}

/** Return the citation chips belonging to one claim. */
export function citationsForClaim(
  evidenceIds: string[],
  citations: AnswerCitation[],
): AnswerCitation[] {
  const byId = new Map(citations.map((item) => [item.evidence_id, item]));
  const found: AnswerCitation[] = [];
  for (const id of evidenceIds) {
    const citation = byId.get(id);
    if (citation) {
      found.push(citation);
    }
  }
  return found;
}
