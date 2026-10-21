import type { ProcessingStatus, QualityStatus } from "@/lib/types";
import {
  processingTone,
  qualityLabel,
  qualityTone,
  stageLabel,
} from "@/lib/format";

const TONES = {
  good: "border-emerald-200 bg-emerald-50 text-emerald-800",
  warn: "border-amber-200 bg-amber-50 text-amber-900",
  bad: "border-red-200 bg-red-50 text-red-800",
  neutral: "border-line bg-background text-muted",
  busy: "border-accent/20 bg-accent-soft text-accent",
} as const;

type Tone = keyof typeof TONES;

/** A small coloured pill. */
export function Badge({
  tone,
  children,
  title,
}: {
  tone: Tone;
  children: React.ReactNode;
  title?: string;
}) {
  return (
    <span
      title={title}
      className={`inline-flex items-center gap-1.5 whitespace-nowrap rounded-full border px-2.5 py-1 text-[11px] font-semibold ${TONES[tone]}`}
    >
      {children}
    </span>
  );
}

/** The badge that shows the processing stage of a document. */
export function ProcessingBadge({ status }: { status: ProcessingStatus }) {
  const tone = processingTone(status);
  return (
    <Badge tone={tone} title={stageLabel(status)}>
      {tone === "busy" ? (
        <span
          aria-hidden="true"
          className="size-1.5 animate-pulse rounded-full bg-current"
        />
      ) : null}
      {stageLabel(status)}
    </Badge>
  );
}

/** The badge that shows the quality of a document. */
export function QualityBadge({ status }: { status: QualityStatus }) {
  return (
    <Badge tone={qualityTone(status)} title={qualityLabel(status)}>
      {qualityLabel(status)}
    </Badge>
  );
}
