/**
 * Plain language messages for the errors the backend can return.
 *
 * This module is safe to use in the browser: it holds no server value and
 * makes no request.
 */

/**
 * The backend error codes the interface explains in plain language.
 *
 * The backend returns a stable code with every failure. The interface shows
 * the message below instead of a raw code, so a lawyer never has to decode
 * an internal name.
 */
export const FRIENDLY_MESSAGES: Record<string, string> = {
  UNSUPPORTED_FILE:
    "That file type is not supported. Please upload a PDF, a Word document, a text file, or an image.",
  FILE_TOO_LARGE:
    "That file is too large to upload. Please split it into smaller parts and try again.",
  PASSWORD_REQUIRED:
    "This PDF is password protected. Please upload a copy that opens without a password.",
  PROCESSING_FAILED:
    "We could not read that file. Please check that it opens correctly and try again.",
  DOCUMENT_NOT_FOUND: "We could not find that document.",
  CASE_NOT_FOUND: "We could not find that case.",
  JOB_NOT_FOUND: "We could not find that job.",
  RATE_LIMITED:
    "Too many requests in a short time. Please wait a moment and try again.",
  UNAUTHORIZED: "Your session has expired. Please sign in again.",
  FORBIDDEN: "You do not have access to that.",
  NOT_FOUND: "We could not find that.",
  SERVICE_UNAVAILABLE:
    "The service is temporarily unavailable. Please try again in a moment.",
  TIMEOUT:
    "The service is taking too long to respond. Please try again in a moment.",
};

const STATUS_MESSAGES: Record<number, string> = {
  400: "That request could not be understood.",
  401: FRIENDLY_MESSAGES.UNAUTHORIZED,
  403: FRIENDLY_MESSAGES.FORBIDDEN,
  404: FRIENDLY_MESSAGES.NOT_FOUND,
  409: "That conflicts with something that already exists.",
  413: FRIENDLY_MESSAGES.FILE_TOO_LARGE,
  415: FRIENDLY_MESSAGES.UNSUPPORTED_FILE,
  422: FRIENDLY_MESSAGES.PROCESSING_FAILED,
  429: FRIENDLY_MESSAGES.RATE_LIMITED,
  500: "Something went wrong on our side. Please try again.",
  503: FRIENDLY_MESSAGES.SERVICE_UNAVAILABLE,
  504: FRIENDLY_MESSAGES.TIMEOUT,
};

const DEFAULT_MESSAGE =
  "Something went wrong. Please try again, and tell us if it keeps happening.";

/** Return the plain language message for a backend error code. */
export function friendlyMessage(
  code: string | undefined,
  fallback?: string,
): string {
  if (code && FRIENDLY_MESSAGES[code]) {
    return FRIENDLY_MESSAGES[code];
  }
  return fallback ?? DEFAULT_MESSAGE;
}

/** Return the plain language message for an HTTP status. */
export function messageForStatus(status: number, detail?: string): string {
  const base = STATUS_MESSAGES[status] ?? DEFAULT_MESSAGE;
  if (detail && detail.trim().length > 0 && !detail.includes("Internal")) {
    return detail;
  }
  return base;
}
