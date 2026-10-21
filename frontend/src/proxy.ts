import { NextResponse, type NextRequest } from "next/server";

import { updateSession } from "@/lib/supabase/proxy";

/** Routes a signed-out visitor may still reach. */
const PUBLIC_ROUTES = ["/login", "/signup"];

/**
 * Protect every route except the public auth pages.
 *
 * A signed-out visitor is sent to the login page. A signed-in visitor who
 * opens the login page is sent to the case list. The proxy refreshes the
 * session on the way through.
 */
export async function proxy(request: NextRequest) {
  let user: { id: string } | null = null;
  let response: NextResponse | null = null;

  try {
    const refreshed = await updateSession(request);
    user = refreshed.user;
    response = refreshed.response;
  } catch {
    // The auth provider could not be reached. Treat the visitor as signed
    // out rather than showing a broken page.
    user = null;
    response = null;
  }

  const { pathname } = request.nextUrl;

  // The API routes answer with JSON. They check the session themselves, so
  // they must never be redirected to the login page.
  if (pathname.startsWith("/api")) {
    return response ?? NextResponse.next({ request });
  }

  const isPublicRoute = PUBLIC_ROUTES.some(
    (route) => pathname === route || pathname.startsWith(`${route}/`),
  );

  if (!user && !isPublicRoute) {
    const loginUrl = request.nextUrl.clone();
    loginUrl.pathname = "/login";
    loginUrl.searchParams.set("next", pathname);
    return NextResponse.redirect(loginUrl);
  }

  if (user && isPublicRoute) {
    const casesUrl = request.nextUrl.clone();
    casesUrl.pathname = "/cases";
    casesUrl.search = "";
    return NextResponse.redirect(casesUrl);
  }

  return response ?? NextResponse.next({ request });
}

export const config = {
  matcher: [
    /*
     * Everything except static assets and image files.
     */
    "/((?!_next/static|_next/image|favicon.ico|.*\\.(?:svg|png|jpg|jpeg|gif|webp)$).*)",
  ],
};
