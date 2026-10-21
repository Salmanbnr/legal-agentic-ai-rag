"use client";

import { useRouter } from "next/navigation";
import { useState, type FormEvent } from "react";

import { createClient } from "@/lib/supabase/client";

/** The account creation form. */
function SignupForm() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    setNotice(null);

    if (password.length < 8) {
      setError("Please choose a password of at least 8 characters.");
      setBusy(false);
      return;
    }

    const supabase = createClient();
    const { data, error: signUpError } = await supabase.auth.signUp({
      email: email.trim(),
      password,
    });

    if (signUpError) {
      setError(
        signUpError.message.toLowerCase().includes("already")
          ? "An account already uses that email address."
          : "We could not create the account. Please try again.",
      );
      setBusy(false);
      return;
    }

    if (!data.session) {
      setNotice(
        "Please check your email to confirm the account, then sign in.",
      );
      setBusy(false);
      return;
    }

    router.replace("/cases");
    router.refresh();
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-5" noValidate>
      <div>
        <label
          htmlFor="email"
          className="block text-sm font-medium text-foreground"
        >
          Email address
        </label>
        <input
          id="email"
          name="email"
          type="email"
          autoComplete="email"
          required
          value={email}
          onChange={(event) => setEmail(event.target.value)}
          className="mt-2 w-full rounded-xl border border-line bg-surface px-4 py-3 text-sm text-foreground outline-none transition focus:border-accent focus:ring-2 focus:ring-accent/20"
          placeholder="you@example.com"
        />
      </div>

      <div>
        <label
          htmlFor="password"
          className="block text-sm font-medium text-foreground"
        >
          Password
        </label>
        <input
          id="password"
          name="password"
          type="password"
          autoComplete="new-password"
          required
          minLength={8}
          value={password}
          onChange={(event) => setPassword(event.target.value)}
          className="mt-2 w-full rounded-xl border border-line bg-surface px-4 py-3 text-sm text-foreground outline-none transition focus:border-accent focus:ring-2 focus:ring-accent/20"
          placeholder="At least 8 characters"
        />
        <p className="mt-2 text-xs text-muted">
          Use at least 8 characters. This account is the only key to your case
          room.
        </p>
      </div>

      {error ? (
        <p
          role="alert"
          className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800"
        >
          {error}
        </p>
      ) : null}

      {notice ? (
        <p
          role="status"
          className="rounded-xl border border-accent/20 bg-accent-soft px-4 py-3 text-sm text-accent"
        >
          {notice}
        </p>
      ) : null}

      <button
        type="submit"
        disabled={busy}
        className="w-full rounded-xl bg-ink px-5 py-3 text-sm font-medium text-surface transition hover:bg-accent disabled:cursor-not-allowed disabled:opacity-60"
      >
        {busy ? "Creating the account…" : "Create account"}
      </button>
    </form>
  );
}

export default function SignupPage() {
  return (
    <main className="flex min-h-screen items-center justify-center px-6 py-12">
      <div className="w-full max-w-sm">
        <div className="mb-8 text-center">
          <div className="mx-auto flex size-11 items-center justify-center rounded-xl bg-ink font-serif text-xl text-surface">
            §
          </div>
          <h1 className="mt-5 font-serif text-3xl tracking-tight">
            Create your account
          </h1>
          <p className="mt-2 text-sm text-muted">
            One account, one private case room.
          </p>
        </div>

        <div className="rounded-2xl border border-line bg-surface p-6 shadow-[0_20px_60px_-40px_rgba(19,32,25,0.5)]">
          <SignupForm />
        </div>

        <p className="mt-6 text-center text-sm text-muted">
          Already have an account?{" "}
          <a href="/login" className="font-medium text-accent underline">
            Sign in
          </a>
        </p>
      </div>
    </main>
  );
}
