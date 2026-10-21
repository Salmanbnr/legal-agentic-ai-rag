import type { ProcessingStatus, QualityStatus } from "@/lib/types";

/** Return a readable file size. */
export function formatBytes(bytes: number | null | undefined): string {
  if (!bytes || bytes < 0) {
    return "—";
  }
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  const units = ["KB", "MB", "GB"];
  let value = bytes / 1024;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  return `${value.toFixed(value >= 10 ? 0 : 1)} ${units[unitIndex]}`;
}

/** Return a readable date. */
export function formatDate(value: string | null | undefined): string {
  if (!value) {
    return "—";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "—";
  }
  return new Intl.DateTimeFormat("en-GB", {
    day: "numeric",
    month: "short",
    year: "numeric",
  }).format(date);
}

/** Return the plain language name of a file type. */
export function formatFileType(mime: string | null | undefined): string {
  if (!mime) {
    return "File";
  }
  const known: Record<string, string> = {
    "application/pdf": "PDF",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
      "Word",
    "application/msword": "Word",
    "text/plain": "Text",
    "image/png": "Image",
    "image/jpeg": "Image",
    "image/tiff": "Image",
  };
  return known[mime] ?? mime.split("/").pop()?.toUpperCase() ?? "File";
}

/** Return the plain language stage a document is at. */
export function stageLabel(status: ProcessingStatus): string {
  const labels: Record<ProcessingStatus, string> = {
    uploaded: "Waiting to start",
    validating: "Checking the file",
    extracting: "Reading the text",
    ocr: "Reading scanned pages",
    chunking: "Splitting into passages",
    embedding: "Preparing the search index",
    indexing: "Adding to the search index",
    ready: "Ready",
    needs_review: "Needs review",
    failed: "Could not be processed",
  };
  return labels[status] ?? "Working";
}

/** Return the plain language quality label. */
export function qualityLabel(status: QualityStatus): string {
  const labels: Record<QualityStatus, string> = {
    pending: "Not checked yet",
    ready: "Good",
    ready_with_warnings: "Usable with warnings",
    needs_review: "Needs review",
    failed: "Could not be read",
    password_required: "Password protected",
  };
  return labels[status] ?? "Unknown";
}

/** Return the tone used for a quality badge. */
export function qualityTone(
  status: QualityStatus,
): "good" | "warn" | "bad" | "neutral" {
  if (status === "ready") {
    return "good";
  }
  if (status === "ready_with_warnings" || status === "password_required") {
    return "warn";
  }
  if (status === "needs_review" || status === "failed") {
    return "bad";
  }
  return "neutral";
}

/** Return the tone used for a processing badge. */
export function processingTone(
  status: ProcessingStatus,
): "good" | "warn" | "bad" | "neutral" | "busy" {
  if (status === "ready") {
    return "good";
  }
  if (status === "needs_review") {
    return "warn";
  }
  if (status === "failed") {
    return "bad";
  }
  if (status === "uploaded") {
    return "neutral";
  }
  return "busy";
}

/** Return the page numbers as a short readable list. */
export function formatPageNumbers(pages: number[]): string {
  if (pages.length === 0) {
    return "";
  }
  if (pages.length <= 6) {
    return pages.join(", ");
  }
  return `${pages.slice(0, 5).join(", ")} and ${pages.length - 5} more`;
}
