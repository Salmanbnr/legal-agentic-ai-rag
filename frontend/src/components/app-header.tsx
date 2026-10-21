import Link from "next/link";

import { SignOutButton } from "@/components/sign-out-button";

type AppHeaderProps = {
  email?: string | null;
};

/** The header shown on every signed-in page. */
export function AppHeader({ email }: AppHeaderProps) {
  return (
    <header className="border-b border-line/80 bg-background/75 backdrop-blur-md">
      <div className="mx-auto flex w-full max-w-6xl items-center justify-between gap-4 px-6 py-5">
        <Link
          href="/cases"
          className="flex items-center gap-3 rounded-lg focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-accent"
        >
          <span className="flex size-10 items-center justify-center rounded-xl bg-ink font-serif text-xl text-surface shadow-sm">
            §
          </span>
          <span>
            <span className="block font-serif text-lg leading-none tracking-tight">
              Legal Evidence Workspace
            </span>
            <span className="mt-1.5 block text-[10px] font-medium uppercase tracking-[0.22em] text-muted">
              Private case room
            </span>
          </span>
        </Link>

        <div className="flex items-center gap-3">
          {email ? (
            <span className="hidden text-xs text-muted sm:inline">{email}</span>
          ) : null}
          <SignOutButton />
        </div>
      </div>
    </header>
  );
}
