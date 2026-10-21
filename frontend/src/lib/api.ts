import "server-only";

import { NextResponse } from "next/server";

import { BackendError, backendFetch } from "@/lib/backend";
import { getSession } from "@/lib/session";

/** Return the request id to use, keeping the one the browser sent. */
export function requestIdFrom(request: Request): string {
  return request.headers.get("x-request-id") ?? crypto.randomUUID();
}

function jsonWithId(body: unknown, status: number, requestId: string) {
  return NextResponse.json(body, {
    status,
    headers: { "X-Request-ID": requestId },
  });
}

/** Return the 401 response used when there is no signed-in user. */
export function unauthorized(requestId: string) {
  return jsonWithId(
    {
      error: "UNAUTHORIZED",
      message: "Your session has expired. Please sign in again.",
      request_id: requestId,
    },
    401,
    requestId,
  );
}

/** Turn any failure into a safe JSON response. */
export function errorResponse(error: unknown, requestId: string) {
  if (error instanceof BackendError) {
    return jsonWithId(
      {
        error: error.code,
        message: error.message,
        request_id: error.requestId ?? requestId,
      },
      error.status,
      error.requestId ?? requestId,
    );
  }
  return jsonWithId(
    {
      error: "INTERNAL_ERROR",
      message:
        "Something went wrong on our side. Please try again, and tell us if it keeps happening.",
      request_id: requestId,
    },
    500,
    requestId,
  );
}

/** The context every proxy route needs. */
export type ApiContext = {
  token: string | null;
  requestId: string;
};

/**
 * Run a proxy route with the signed-in user and the request id.
 *
 * The browser only ever calls these routes. The Python address and the
 * access token stay on the server.
 */
export async function withSession<T>(
  request: Request,
  handler: (context: ApiContext) => Promise<T>,
): Promise<Response> {
  const requestId = requestIdFrom(request);
  const session = await getSession();
  if (!session) {
    return unauthorized(requestId);
  }

  try {
    const data = await handler({
      token: session.token,
      requestId,
    });
    if (data instanceof Response) {
      return data;
    }
    return jsonWithId(data, 200, requestId);
  } catch (error) {
    return errorResponse(error, requestId);
  }
}

/** Call the backend on behalf of the signed-in user. */
export function callBackend<T>(
  path: string,
  context: ApiContext,
  options: {
    method?: "GET" | "POST" | "PATCH" | "DELETE";
    body?: BodyInit | null;
    headers?: Record<string, string>;
    timeoutMs?: number;
  } = {},
): Promise<T> {
  return backendFetch<T>(path, {
    method: options.method ?? "GET",
    token: context.token,
    body: options.body ?? null,
    headers: options.headers,
    requestId: context.requestId,
    timeoutMs: options.timeoutMs,
  });
}
