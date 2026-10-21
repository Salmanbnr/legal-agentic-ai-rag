import "server-only";

import { friendlyMessage, messageForStatus } from "@/lib/messages";

export { friendlyMessage, messageForStatus };

/** A failed backend call with a message that is safe to show. */
export class BackendError extends Error {
  readonly status: number;
  readonly code: string;
  readonly requestId?: string;

  constructor(
    status: number,
    code: string,
    message: string,
    requestId?: string,
  ) {
    super(message);
    this.name = "BackendError";
    this.status = status;
    this.code = code;
    this.requestId = requestId;
  }
}

function backendUrl() {
  const base = process.env.AI_BACKEND_BASE_URL;
  if (!base) {
    throw new Error("AI_BACKEND_BASE_URL is not configured");
  }
  return base.replace(/\/+$/, "");
}

type BackendFetchOptions = {
  method?: "GET" | "POST" | "PATCH" | "DELETE";
  token?: string | null;
  body?: BodyInit | null;
  headers?: Record<string, string>;
  requestId?: string | null;
  signal?: AbortSignal;
  timeoutMs?: number;
};

/**
 * How long the server waits for the backend before giving up.
 *
 * Without a limit a wedged backend leaves the page spinning forever, because
 * the server component never finishes rendering. Every call therefore has a
 * deadline, and a long one is only granted where the work is known to be slow.
 */
export const BACKEND_TIMEOUT_MS = readTimeout(
  "AI_BACKEND_TIMEOUT_MS",
  30_000,
);
export const BACKEND_UPLOAD_TIMEOUT_MS = readTimeout(
  "AI_BACKEND_UPLOAD_TIMEOUT_MS",
  180_000,
);
export const BACKEND_ANSWER_TIMEOUT_MS = readTimeout(
  "AI_BACKEND_ANSWER_TIMEOUT_MS",
  120_000,
);

function readTimeout(name: string, fallback: number): number {
  const raw = Number(process.env[name]);
  return Number.isFinite(raw) && raw > 0 ? raw : fallback;
}

function isTimeoutError(error: unknown): boolean {
  if (!(error instanceof Error)) {
    return false;
  }
  return error.name === "TimeoutError" || error.name === "AbortError";
}

/**
 * Call the Python API from the server.
 *
 * The user access token is forwarded as a Bearer header so the backend can
 * check ownership. The backend address never reaches the browser: this module
 * is server only.
 */
export async function backendFetch<T>(
  path: string,
  options: BackendFetchOptions = {},
): Promise<T> {
  const {
    method = "GET",
    token,
    body,
    headers = {},
    requestId,
    signal,
    timeoutMs = BACKEND_TIMEOUT_MS,
  } = options;

  const requestHeaders: Record<string, string> = {
    Accept: "application/json",
    ...headers,
  };
  if (token) {
    requestHeaders.Authorization = `Bearer ${token}`;
  }
  if (requestId) {
    requestHeaders["X-Request-ID"] = requestId;
  }
  if (body && !(body instanceof FormData) && !requestHeaders["Content-Type"]) {
    requestHeaders["Content-Type"] = "application/json";
  }

  let response: Response;
  try {
    response = await fetch(`${backendUrl()}${path}`, {
      method,
      headers: requestHeaders,
      body: body ?? undefined,
      cache: "no-store",
      signal: signal ?? AbortSignal.timeout(timeoutMs),
    });
  } catch (error) {
    if (isTimeoutError(error) && !signal?.aborted) {
      throw new BackendError(
        504,
        "TIMEOUT",
        friendlyMessage("TIMEOUT"),
        requestId ?? undefined,
      );
    }
    throw new BackendError(
      503,
      "SERVICE_UNAVAILABLE",
      friendlyMessage("SERVICE_UNAVAILABLE"),
      requestId ?? undefined,
    );
  }

  const responseRequestId =
    response.headers.get("x-request-id") ?? requestId ?? undefined;

  if (!response.ok) {
    const detail = await readErrorDetail(response);
    throw new BackendError(
      response.status,
      `HTTP_${response.status}`,
      messageForStatus(response.status, detail),
      responseRequestId,
    );
  }

  if (response.status === 204) {
    return undefined as T;
  }
  return (await response.json()) as T;
}

async function readErrorDetail(response: Response): Promise<string | undefined> {
  try {
    const body = (await response.json()) as { detail?: unknown };
    return typeof body.detail === "string" ? body.detail : undefined;
  } catch {
    return undefined;
  }
}
