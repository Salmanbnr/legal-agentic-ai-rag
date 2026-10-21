"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

/** The two areas of a case: the documents and the chat. */
const TABS = [
  { href: "/cases", label: "Documents" },
  { href: "/chat", label: "Chat" },
] as const;

type CaseTabsProps = {
  caseId: string;
};

/** The tab bar of a case. */
export function CaseTabs({ caseId }: CaseTabsProps) {
  const pathname = usePathname();
  const base = `/cases/${encodeURIComponent(caseId)}`;

  return (
    <nav aria-label="Case sections">
      <ul className="flex gap-2">
        {TABS.map((tab) => {
          const href = `${base}${tab.href === "/cases" ? "" : tab.href}`;
          const active = pathname === href;
          return (
            <li key={tab.href}>
              <Link
                href={href}
                aria-current={active ? "page" : undefined}
                className={`inline-flex rounded-full px-4 py-2 text-sm font-medium transition ${
                  active
                    ? "bg-ink text-surface"
                    : "border border-line bg-surface/70 text-muted hover:text-foreground"
                }`}
              >
                {tab.label}
              </Link>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}
