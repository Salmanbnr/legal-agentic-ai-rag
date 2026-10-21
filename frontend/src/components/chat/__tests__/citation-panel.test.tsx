import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { CitationPanel } from "@/components/chat/citation-panel";
import type { EvidenceDetail } from "@/lib/types";
import { makeCitation } from "./fixtures";

const DETAIL: EvidenceDetail = {
  evidence_id: "ev-1",
  case_id: "case-1",
  document_id: "doc-1",
  document_title: "Lease agreement.pdf",
  document_mime: "application/pdf",
  page_start: 7,
  page_end: 8,
  section_path: ["Clause 12"],
  text: "The monthly rent is GBP 1,250, payable on the first day of each month.",
  extraction_method: "ocr",
  ocr_confidence: 58,
  quality_status: "ready_with_warnings",
  quality_flags: ["low_ocr_confidence"],
  quality_warnings: [
    "Low OCR confidence on this page: a name, date, or amount may have been misread.",
  ],
  source_anchor: "doc:1|p:7|c:0",
};

const fetchMock = vi.fn();

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

beforeEach(() => {
  fetchMock.mockReset();
  vi.stubGlobal("fetch", fetchMock);
  vi.spyOn(window, "open").mockImplementation(() => null);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("CitationPanel", () => {
  it("shows the stored text, page range, extraction method, and warning", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse(DETAIL));

    render(
      <CitationPanel
        caseId="case-1"
        citation={makeCitation()}
        onClose={vi.fn()}
      />,
    );

    expect(await screen.findByText(DETAIL.text)).toBeInTheDocument();
    expect(screen.getByRole("dialog")).toHaveAccessibleName(
      "Lease agreement.pdf",
    );
    expect(screen.getAllByText("pp. 7–8").length).toBeGreaterThan(0);
    expect(screen.getByText("Read with OCR from a scan")).toBeInTheDocument();
    expect(
      screen.getAllByText("Low OCR confidence on this page", {
        exact: false,
      }).length,
    ).toBeGreaterThan(0);
    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining("/api/cases/case-1/evidence/ev-1"),
      expect.anything(),
    );
  });

  it("opens the original through a fresh signed link at the cited page", async () => {
    const user = userEvent.setup();
    fetchMock
      .mockResolvedValueOnce(jsonResponse(DETAIL))
      .mockResolvedValueOnce(
        jsonResponse({
          document_id: "doc-1",
          filename: "Lease agreement.pdf",
          mime: "application/pdf",
          url: "https://example.test/signed",
          expires_in_seconds: 60,
        }),
      );

    render(
      <CitationPanel
        caseId="case-1"
        citation={makeCitation()}
        onClose={vi.fn()}
      />,
    );
    await screen.findByText(DETAIL.text);

    await user.click(screen.getByRole("button", { name: "Open original" }));

    await waitFor(() =>
      expect(window.open).toHaveBeenCalledWith(
        "https://example.test/signed#page=7",
        "_blank",
        "noopener,noreferrer",
      ),
    );
    expect(fetchMock).toHaveBeenLastCalledWith(
      expect.stringContaining("/api/documents/doc-1/signed-url"),
      expect.anything(),
    );
  });

  it("shows a safe message when the evidence cannot be loaded", async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({ message: "We could not find that document." }, 404),
    );

    render(
      <CitationPanel
        caseId="case-1"
        citation={makeCitation()}
        onClose={vi.fn()}
      />,
    );

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("We could not find that document.");
    expect(
      screen.getByRole("button", { name: "Try again" }),
    ).toBeInTheDocument();
    expect(screen.queryByText(DETAIL.text)).not.toBeInTheDocument();
  });

  it("reports a signed-link failure without technical details", async () => {
    const user = userEvent.setup();
    fetchMock
      .mockResolvedValueOnce(jsonResponse(DETAIL))
      .mockResolvedValueOnce(
        jsonResponse(
          { message: "The original file is temporarily unavailable" },
          503,
        ),
      );

    render(
      <CitationPanel
        caseId="case-1"
        citation={makeCitation()}
        onClose={vi.fn()}
      />,
    );
    await screen.findByText(DETAIL.text);

    await user.click(screen.getByRole("button", { name: "Open original" }));

    expect(
      await screen.findByText("The original file is temporarily unavailable"),
    ).toBeInTheDocument();
    expect(window.open).not.toHaveBeenCalled();
  });

  it("closes on Escape", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    fetchMock.mockResolvedValueOnce(jsonResponse(DETAIL));

    render(
      <CitationPanel
        caseId="case-1"
        citation={makeCitation()}
        onClose={onClose}
      />,
    );
    await screen.findByText(DETAIL.text);

    await user.keyboard("{Escape}");

    expect(onClose).toHaveBeenCalled();
  });
});
