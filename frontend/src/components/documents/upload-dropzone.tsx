"use client";

import { useCallback, useRef, useState, type DragEvent } from "react";

import { friendlyMessage } from "@/lib/messages";
import type { UploadResponse } from "@/lib/types";

type UploadState = {
  id: string;
  name: string;
  progress: number;
  status: "uploading" | "done" | "error";
  message?: string;
};

/** Long enough for a large file, short enough to release a stuck screen. */
const UPLOAD_CLIENT_TIMEOUT_MS = 190_000;

type UploadDropzoneProps = {
  caseId: string;
  onUploaded?: (jobId: string | null) => void;
};

/**
 * The drag and drop upload area.
 *
 * Files are sent one at a time so the progress bar shows the real progress of
 * the file being uploaded. The browser only talks to this app's own routes;
 * the backend address is never exposed here.
 */
export function UploadDropzone({ caseId, onUploaded }: UploadDropzoneProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [uploads, setUploads] = useState<UploadState[]>([]);

  const update = useCallback(
    (id: string, changes: Partial<UploadState>) => {
      setUploads((current) =>
        current.map((item) => (item.id === id ? { ...item, ...changes } : item)),
      );
    },
    [],
  );

  const upload = useCallback(
    (file: File) =>
      new Promise<void>((resolve) => {
        const id = `${file.name}-${file.size}-${Date.now()}`;
        setUploads((current) => [
          ...current,
          { id, name: file.name, progress: 0, status: "uploading" },
        ]);

        const body = new FormData();
        body.append("file", file, file.name);

        const request = new XMLHttpRequest();
        request.open("POST", `/api/cases/${encodeURIComponent(caseId)}/documents`);
        request.timeout = UPLOAD_CLIENT_TIMEOUT_MS;

        request.upload.addEventListener("progress", (event) => {
          if (event.lengthComputable) {
            const percent = Math.round((event.loaded / event.total) * 100);
            update(id, { progress: percent });
          }
        });

        request.addEventListener("load", () => {
          let payload: unknown = null;
          try {
            payload = JSON.parse(request.responseText);
          } catch {
            payload = null;
          }

          if (request.status >= 200 && request.status < 300) {
            const results = (Array.isArray(payload) ? payload : [payload]) as
              | UploadResponse[]
              | null;
            const first = results?.[0];
            if (first?.error_code) {
              update(id, {
                status: "error",
                progress: 100,
                message: friendlyMessage(first.error_code),
              });
              onUploaded?.(null);
            } else {
              update(id, { status: "done", progress: 100 });
              onUploaded?.(first?.job_id ?? null);
            }
          } else {
            const message = (payload as { message?: string } | null)?.message;
            update(id, {
              status: "error",
              progress: 100,
              message:
                message ??
                "We could not upload that file. Please try again.",
            });
            onUploaded?.(null);
          }
          resolve();
        });

        request.addEventListener("error", () => {
          update(id, {
            status: "error",
            progress: 100,
            message:
              "We could not reach the service. Please check your connection and try again.",
          });
          onUploaded?.(null);
          resolve();
        });

        request.addEventListener("timeout", () => {
          update(id, {
            status: "error",
            progress: 100,
            message:
              "The upload took too long. Please try again, and tell us if it keeps happening.",
          });
          onUploaded?.(null);
          resolve();
        });

        request.send(body);
      }),
    [caseId, onUploaded, update],
  );

  const uploadFiles = useCallback(
    async (files: File[]) => {
      for (const file of files) {
        await upload(file);
      }
    },
    [upload],
  );

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragging(false);
    const files = Array.from(event.dataTransfer.files ?? []);
    if (files.length > 0) {
      void uploadFiles(files);
    }
  }

  const busy = uploads.some((item) => item.status === "uploading");

  return (
    <div className="space-y-4">
      <div
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        className={`rounded-2xl border border-dashed px-6 py-10 text-center transition ${
          dragging
            ? "border-accent bg-accent-soft/60"
            : "border-line bg-surface/70"
        }`}
      >
        <p className="font-serif text-xl">Drop the case files here</p>
        <p className="mt-1.5 text-sm text-muted">
          PDF, Word, text, or images. You can add more than one file. The
          originals are never changed.
        </p>
        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          disabled={busy}
          className="mt-5 rounded-full bg-ink px-5 py-2.5 text-sm font-medium text-surface transition hover:bg-accent disabled:opacity-60"
        >
          {busy ? "Uploading…" : "Choose files"}
        </button>
        <input
          ref={inputRef}
          type="file"
          multiple
          className="sr-only"
          aria-label="Choose case files to upload"
          onChange={(event) => {
            const files = Array.from(event.target.files ?? []);
            if (files.length > 0) {
              void uploadFiles(files);
            }
            event.target.value = "";
          }}
        />
      </div>

      {uploads.length > 0 ? (
        <ul className="space-y-3">
          {uploads.map((item) => (
            <li
              key={item.id}
              className="rounded-xl border border-line bg-surface px-4 py-3"
            >
              <div className="flex items-center justify-between gap-4">
                <p className="truncate text-sm font-medium">{item.name}</p>
                <p className="shrink-0 text-xs text-muted">
                  {item.status === "uploading"
                    ? `${item.progress}%`
                    : item.status === "done"
                      ? "Uploaded"
                      : "Not uploaded"}
                </p>
              </div>
              <div
                role="progressbar"
                aria-label={`Uploading ${item.name}`}
                aria-valuenow={item.progress}
                aria-valuemin={0}
                aria-valuemax={100}
                className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-line"
              >
                <div
                  className={`h-full rounded-full transition-all ${
                    item.status === "error" ? "bg-red-500" : "bg-accent"
                  }`}
                  style={{ width: `${item.progress}%` }}
                />
              </div>
              {item.message ? (
                <p role="alert" className="mt-2 text-xs text-red-700">
                  {item.message}
                </p>
              ) : null}
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}
