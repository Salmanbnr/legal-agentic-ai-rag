"use client";

import { createBrowserClient } from "@supabase/ssr";

/**
 * Return a Supabase client for the browser.
 *
 * Only the publishable key is used. It is designed to be public; the service
 * role key and the backend URL never reach the browser.
 */
export function createClient() {
  return createBrowserClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL ?? "",
    process.env.NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY ?? "",
  );
}
