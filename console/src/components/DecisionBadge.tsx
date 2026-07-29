import type { Decision } from "@/api/queries";

const INK: Record<Decision, string> = {
  allow: "text-ink border-ink/60 dark:text-ink-dark dark:border-ink-dark/50",
  mask: "text-mask border-mask dark:text-mask-dark dark:border-mask-dark",
  block: "text-block border-block dark:text-block-dark dark:border-block-dark",
};

// A slight, fixed per-decision rotation and asymmetric corner radius — a
// real stamp presses unevenly onto paper. This is the one deliberately
// irregular shape in an otherwise rectilinear system (see DESIGN.md
// Shapes). Rotation is tiny and fixed (not random per render) so it reads
// as "this stamp" rather than jittering between re-renders.
const TILT: Record<Decision, string> = {
  allow: "-rotate-1",
  mask: "rotate-1",
  block: "-rotate-2",
};

export function DecisionBadge({
  decision,
  animateIn,
}: {
  decision: Decision;
  /** Plays the press-down/settle landing — reserved for the dry-run
   * recorder strip's signature moment (see DESIGN.md's one shadow
   * exception beyond overlays). Not used for badges in tables/lists. */
  animateIn?: boolean;
}) {
  return (
    <span className={`inline-block ${TILT[decision]}`}>
      <span
        className={`inline-block px-2 py-0.5 font-mono text-xs font-semibold tracking-wider uppercase ${INK[decision]} ${animateIn ? "stamp-land" : ""}`}
        style={{
          borderRadius: "3px 7px 4px 8px",
          // Uneven border weight per side — a stamp presses harder on one
          // edge than the other. Paired with the fixed per-decision
          // rotation above, not random, so it reads as "this stamp," not
          // jitter between renders.
          borderStyle: "solid",
          borderWidth: "2.5px 1.5px 2px 2px",
        }}
      >
        {decision}
      </span>
    </span>
  );
}
