import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { AnswerCard } from "@/components/chat/answer-card";
import { makeCitation, makeRecord, makeTrace } from "./fixtures";

const DIRECT_CLAIM = {
  text: "Clause 12 sets the rent at GBP 1,250 per month.",
  evidence_ids: ["ev-1"],
  type: "direct" as const,
  confidence: null,
};

const INFERENCE_CLAIM = {
  text: "The parties likely intended a yearly review.",
  evidence_ids: ["ev-1"],
  type: "inference" as const,
  confidence: null,
};

beforeEach(() => {
  Object.defineProperty(navigator, "clipboard", {
    configurable: true,
    value: { writeText: vi.fn().mockResolvedValue(undefined) },
  });
});

describe("AnswerCard statuses", () => {
  const cases = [
    { status: "answered", banner: null },
    { status: "partial", banner: null },
    { status: "insufficient_evidence", banner: "Not in the files" },
    { status: "conflicting_evidence", banner: "The documents disagree" },
  ] as const;

  it.each(cases)("renders a $status answer", ({ status, banner }) => {
    const abstained = status === "insufficient_evidence";
    const record = makeRecord({
      final_status: status,
      answer_text: abstained ? "" : "The rent is GBP 1,250 per month.",
      claims: abstained ? [] : [DIRECT_CLAIM],
      evidence: abstained ? [] : [makeCitation()],
      conflicts:
        status === "conflicting_evidence"
          ? ["Clause 12 says GBP 1,250, while the addendum says GBP 1,400."]
          : [],
      limitations: ["The addendum is only partly legible."],
    });

    const { container } = render(
      <AnswerCard record={record} onOpenCitation={vi.fn()} />,
    );

    expect(
      screen.getByRole("heading", { name: "Short answer" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: "Supporting evidence" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: "Limitations" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: "Suggested follow-up" }),
    ).toBeInTheDocument();
    expect(container.querySelector(`[data-status="${status}"]`)).not.toBeNull();

    if (banner) {
      expect(screen.getByText(banner)).toBeInTheDocument();
    } else {
      expect(screen.queryByText("Not in the files")).not.toBeInTheDocument();
      expect(
        screen.queryByText("The documents disagree"),
      ).not.toBeInTheDocument();
    }

    if (abstained) {
      expect(
        screen.getByText("No page in the case file supports an answer to this question."),
      ).toBeInTheDocument();
    }

    if (status === "conflicting_evidence") {
      expect(screen.getByRole("heading", { name: "Conflicts" })).toBeInTheDocument();
      expect(
        screen.getByText("Clause 12 says GBP 1,250, while the addendum says GBP 1,400."),
      ).toBeInTheDocument();
    } else {
      expect(
        screen.queryByRole("heading", { name: "Conflicts" }),
      ).not.toBeInTheDocument();
    }
  });
});

describe("AnswerCard evidence", () => {
  it("separates direct statements from inference", () => {
    const record = makeRecord({
      claims: [DIRECT_CLAIM, INFERENCE_CLAIM],
    });

    render(<AnswerCard record={record} onOpenCitation={vi.fn()} />);

    expect(screen.getByText("Stated in the documents")).toBeInTheDocument();
    expect(
      screen.getByText(/Inference — the assistant/),
    ).toBeInTheDocument();
    expect(screen.getByText(DIRECT_CLAIM.text)).toBeInTheDocument();
    expect(screen.getByText(INFERENCE_CLAIM.text)).toBeInTheDocument();
  });

  it("labels a citation with the document and page and opens it", async () => {
    const user = userEvent.setup();
    const onOpenCitation = vi.fn();
    const record = makeRecord();

    render(<AnswerCard record={record} onOpenCitation={onOpenCitation} />);

    const chip = screen.getByRole("button", {
      name: "View the evidence from Lease agreement.pdf, p. 7",
    });
    await user.click(chip);

    expect(onOpenCitation).toHaveBeenCalledWith(record.answer.evidence[0]);
  });

  it("copies the answer with its citations in plain text", async () => {
    const record = makeRecord();

    render(<AnswerCard record={record} onOpenCitation={vi.fn()} />);
    fireEvent.click(screen.getByRole("button", { name: "Copy answer" }));

    await waitFor(() =>
      expect(navigator.clipboard.writeText).toHaveBeenCalledTimes(1),
    );
    const copied = (navigator.clipboard.writeText as unknown as {
      mock: { calls: string[][] };
    }).mock.calls[0][0];
    expect(copied).toContain("Question: What is the monthly rent?");
    expect(copied).toContain("Clause 12 sets the rent at GBP 1,250 per month.");
    expect(copied).toContain("[Lease agreement.pdf, p. 7]");
    expect(copied).toContain("Sources:");
    expect(await screen.findByText("Copied with citations")).toBeInTheDocument();
  });
});

describe("AnswerCard workflow panel", () => {
  it("shows the plan, queries, retries, and nodes of an agentic answer", () => {
    const record = makeRecord({}, { mode: "agentic", trace: makeTrace() });

    render(<AnswerCard record={record} onOpenCitation={vi.fn()} />);

    expect(screen.getByText("How this was found")).toBeInTheDocument();
    expect(
      screen.getAllByText("What rent does clause 12 set?").length,
    ).toBeGreaterThan(0);
    expect(screen.getByText(/1 retry/)).toBeInTheDocument();
    expect(screen.getByText("final_gate")).toBeInTheDocument();
  });

  it("does not show the workflow panel for a standard answer", () => {
    render(<AnswerCard record={makeRecord()} onOpenCitation={vi.fn()} />);

    expect(screen.queryByText("How this was found")).not.toBeInTheDocument();
  });
});
