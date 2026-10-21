import "server-only";

import { createClient } from "@/lib/supabase/server";

/**
 * Return the signed-in user and their access token, or null.
 *
 * The token is only ever used on the server to call the backend. It is never
 * sent to the browser.
 */
export async function getSession() {
  const supabase = await createClient();
  const {
    data: { user },
    error,
  } = await supabase.auth.getUser();

  if (error || !user) {
    return null;
  }

  const { data: sessionData } = await supabase.auth.getSession();
  const token = sessionData.session?.access_token ?? null;

  return { user, token, supabase };
}

/** Return the signed-in user id, or null. */
export async function getUserId(): Promise<string | null> {
  const session = await getSession();
  return session?.user.id ?? null;
}
