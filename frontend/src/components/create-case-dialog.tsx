"use client";

import { useRouter } from "next/navigation";
import { useState, type FormEvent } from "react";

import { Dialog } from "@/components/dialog";
import type { CaseSummary } from "@/lib/types";

/** The button and dialog that create a new case. */
export function CreateCaseDialog() {
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy(true);
    setError(null);

    try {
      const response = await fetch("/api/cases", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title, description }),
      });
      const body = (await response.json()) as
        | CaseSummary
        | { message?: string };

      if (!response.ok) {
        setError(
          (body as { message?: string }).message ??
            "We could not create the case. Please try again.",
        );
        setBusy(false);
        return;
      }

      const created = body as CaseSummary;
      setOpen(false);
      setTitle("");
      setDescription("");
      router.push(`/cases/${created.id}`);
      router.refresh();
    } catch {
      setError("We could not reach the service. Please try again.");
      setBusy(false);
    }
  }

  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        className="rounded-full bg-ink px-5 py-2.5 text-sm font-medium text-surface transition hover:bg-accent"
      >
        New case
      </button>

      <Dialog
        open={open}
        title="New case"
        description="A case holds the documents you upload and the answers you get from them."
        onClose={() => setOpen(false)}
        footer={
          <>
            <button
              type="button"
              onClick={() => setOpen(false)}
              className="rounded-full border border-line px-4 py-2 text-sm font-medium text-muted transition hover:text-foreground"
            >
              Cancel
            </button>
            <button
              type="submit"
              form="create-case-form"
              disabled={busy}
              className="rounded-full bg-ink px-4 py-2 text-sm font-medium text-surface transition hover:bg-accent disabled:opacity-60"
            >
              {busy ? "Creating…" : "Create case"}
            </button>
          </>
        }
      >
        <form id="create-case-form" onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label
              htmlFor="case-title"
              className="block text-sm font-medium text-foreground"
            >
              Case name
            </label>
            <input
              id="case-title"
              name="title"
              value={title}
              onChange={(event) => setTitle(event.target.value)}
              required
              maxLength={300}
              className="mt-2 w-full rounded-xl border border-line bg-background px-4 py-3 text-sm outline-none transition focus:border-accent focus:ring-2 focus:ring-accent/20"
              placeholder="Smith v Jones — rent arrears"
            />
          </div>

          <div>
            <label
              htmlFor="case-description"
              className="block text-sm font-medium text-foreground"
            >
              What is this case about? (optional)
            </label>
            <textarea
              id="case-description"
              name="description"
              value={description}
              onChange={(event) => setDescription(event.target.value)}
              rows={3}
              maxLength={20000}
              className="mt-2 w-full resize-y rounded-xl border border-line bg-background px-4 py-3 text-sm outline-none transition focus:border-accent focus:ring-2 focus:ring-accent/20"
              placeholder="Dispute over unpaid rent after a repair was refused."
            />
          </div>

          {error ? (
            <p
              role="alert"
              className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800"
            >
              {error}
            </p>
          ) : null}
        </form>
      </Dialog>
    </>
  );
}
