---
name: SentinelLM Console
description: A seismograph/polygraph instrument room for a live security gateway — ink traces, not SaaS cards.
colors:
  paper: "#f6f1e5"
  paper-dark: "#1b1712"
  ink: "#211c13"
  ink-dark: "#ede6d6"
  ink-muted: "#6b6250"
  ink-muted-dark: "#a69a80"
  hairline: "#ddd4bc"
  hairline-dark: "#3a3226"
  hairline-strong: "#b8ac8c"
  hairline-strong-dark: "#55492f"
  block: "#b7391e"
  block-dark: "#e2602e"
  mask: "#8a5e0a"
  mask-dark: "#d9a438"
typography:
  data:
    fontFamily: "IBM Plex Mono, ui-monospace, SF Mono, Consolas, monospace"
    fontSize: "0.75rem–1.5rem (context-sized; 1.5rem/24px ceiling in MetricCard)"
    fontWeight: 400
    lineHeight: normal
    letterSpacing: normal
  title:
    fontFamily: "-apple-system, SF Pro Text, system-ui, Segoe UI, sans-serif"
    fontSize: "0.875rem"
    fontWeight: 600
    lineHeight: normal
    letterSpacing: "0.15em"
  label:
    fontFamily: "-apple-system, SF Pro Text, system-ui, Segoe UI, sans-serif"
    fontSize: "0.75rem"
    fontWeight: 500
    lineHeight: normal
    letterSpacing: "0.05em–0.2em"
  body:
    fontFamily: "-apple-system, SF Pro Text, system-ui, Segoe UI, sans-serif"
    fontSize: "0.875rem"
    fontWeight: 400
    lineHeight: normal
rounded:
  none: "0"
  stamp: "3px 7px 4px 8px"
spacing:
  xs: "8px"
  sm: "12px"
  md: "16px"
  lg: "24px"
components:
  button-primary:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.paper}"
    rounded: "{rounded.none}"
    padding: "6px 16px"
  button-primary-hover:
    backgroundColor: "transparent"
    textColor: "{colors.ink}"
  input-text:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
    rounded: "{rounded.none}"
    padding: "6px 12px"
  decision-stamp-allow:
    textColor: "{colors.ink}"
    rounded: "{rounded.stamp}"
    padding: "2px 8px"
  decision-stamp-mask:
    textColor: "{colors.mask}"
    rounded: "{rounded.stamp}"
    padding: "2px 8px"
  decision-stamp-block:
    textColor: "{colors.block}"
    rounded: "{rounded.stamp}"
    padding: "2px 8px"
---

# Design System: SentinelLM Console

## Overview

**Creative North Star: "The Instrument Room"**

The console is built as a technician's recording desk: a live seismograph or polygraph, continuously tracing a stream of requests on chart paper, with a pen that marks anomalies in colored ink the instant they happen. The metaphor is not decorative — SentinelLM's actual job is exactly this: a continuous stream of traffic, monitored in real time, with detections flagged the moment they occur. The three built pages (Policy, Audit, Metrics) read as three channels on one recording desk, not sidebar-nav SaaS sections.

Two things were deliberately rejected in choosing this direction, and both rejections held through implementation. First, the indigo-and-slate default this project already had — recognizable on sight as an interchangeable SaaS-admin template. Second, the more obvious "security tool" answer — a green-phosphor, black-background hacker terminal — a costume, not a credible instrument. What actually shipped is lit, clinical, and precise: a warm paper ground, hard black ink structure, and exactly two severity inks, built with Tailwind v4's CSS-first `@theme` (no `tailwind.config.*` file — every token below lives in `console/src/index.css`).

The build confirms the seed's "bolder pass" intent landed correctly: the paper has real material presence (a two-layer chain-line texture, not a flat tint), the two severity inks are saturated stamp-pad colors rather than dusty pastels, and the dry-run panel's trace-draw-and-stamp-land is the one place motion and shadow are allowed to be genuinely theatrical. Everywhere else stayed exactly as quiet as planned.

**Key Characteristics:**
- Warm paper ground with a real two-layer laid-paper texture, not a flat tint — black ink line-work, two severity inks (never more), both saturated enough to have real presence when they appear
- IBM Plex Mono carries every gateway-produced fact (numbers, timestamps, IDs, entity types); the system sans stack carries every label the console itself authored
- Flat by construction — depth comes from a three-tier line hierarchy (hairline / hairline-strong / 2px ink) and a hover ink-wash, never a drop shadow used as decoration, except the drawer overlay and the dry-run stamp-landing instant
- Hard, rectilinear structure everywhere except the one place a real instrument would actually be soft: the ink stamp
- A single quiet kinetic tell — a live "recording" pulse in the nav shell, in graphite ink, never a severity color — so the resting state reads as alive, not dormant
- The type ramp itself enacts the Instrument/Chrome Split: there is no display or headline tier. The largest text in the system (24px) is a mono number in `MetricCard` — gateway-produced data, not console-authored chrome.

## Colors

Restrained strategy: one warm neutral ground, ink-black for structure and the ALLOW state, and exactly two severity inks — each with a `-dark` pair for night-shift mode. No third accent, no gradient, no tint-on-tint SaaS palette. Contrast-checked against WCAG AA (4.5:1) for every color used as text: ink 15.0:1, ink-muted 5.3:1, block 5.15:1, mask 5.05:1 on light paper; ink-dark 14.3:1, ink-muted-dark 6.4:1, block-dark 5.0:1, mask-dark 7.9:1 on dark paper.

### Primary
- **Alert Vermillion** (`#b7391e` / dark-mode `#e2602e`) (BLOCK ink): reserved for blocked decisions, active anomaly marks in redacted text, and the destructive-action register — the delete-confirm affordance in `PolicyTable`, form/mutation error text, `ErrorState`, and `ErrorBoundary`'s crash panel all render in this ink. It is never used as a brand accent, nav color, or link color at rest; the built system draws the line at *severity and failure states*, not at "any use beyond BLOCK badges."

### Secondary
- **Ochre Amber** (`#8a5e0a` / dark-mode `#d9a438`) (MASK ink): the masked/redacted-decision ink. In light mode it renders as a deep, earthy ochre rather than a bright amber — the dark-mode value (`#d9a438`) is the one that reads as amber; both stay clearly distinguishable from vermillion by hue, not just lightness, for common color-vision deficiencies.

### Neutral
- **Chart Paper** (`#f6f1e5` / dark-mode `#1b1712`): the base surface for every screen, carrying a two-layer `repeating-linear-gradient` texture (see Elevation & Depth) rather than a flat tint.
- **Recorder Ink** (`#211c13` / dark-mode `#ede6d6`): structural lines, body text, and the calm/ALLOW state.
- **Ink Muted** (`#6b6250` / dark-mode `#a69a80`): secondary text — labels, sublabels, timestamps, muted metadata.
- **Hairline** (`#ddd4bc` / dark-mode `#3a3226`): the lightest line weight — row dividers inside tables.
- **Hairline Strong** (`#b8ac8c` / dark-mode `#55492f`): container, panel, input, and table borders — one step heavier than a row divider.
- Both severity inks carry an unused `-tint` pair (`--color-block-tint`, `--color-mask-tint` and their `-dark` variants) declared in `index.css` — see the closing line of this document for why they're not canonized here.

### Named Rules
**The Two-Ink Rule.** Only two severity colors exist system-wide: Alert Vermillion for BLOCK, Ochre Amber for MASK. A third accent color is a sign the system has drifted from the instrument metaphor back toward generic dashboard decoration. Severity ink extends to failure/destructive states (form errors, crash boundary, delete-confirm) because those are real BLOCK-adjacent events, not decoration — but it never becomes a permanent at-rest UI accent, brand mark, nav color, or link color. (The `NavBar` recording-pulse dot is graphite `ink`, not `block`, specifically because a resting UI element is not a severity event.)

**The Confident Ink Rule.** A severity color that could be mistaken for a pastel has failed. Rarity of use is the restraint; saturation of the color itself is not where restraint applies.

## Typography

**Data Font:** IBM Plex Mono (self-hosted via `@fontsource/ibm-plex-mono`, weights 400/500/600), falling back to `ui-monospace, "SF Mono", Consolas, monospace` — every number, timestamp, entity type, latency figure, decision word, and ID reads in this face.

**UI/Label Font:** the system sans stack (`-apple-system, "SF Pro Text", system-ui, "Segoe UI", sans-serif`) — chrome, labels, page titles, and navigation.

**Character:** clinical and precise, not playful. The pairing reads like a lab instrument's printed output next to its control-panel labels.

### Hierarchy
There is no display or headline tier — the built system's largest text is data, not chrome:
- **Data** (regular/semibold, mono, 0.75rem–1.5rem depending on context; 1.5rem/24px is the ceiling, used only for the `MetricCard` headline number): every timestamp, ID, latency figure, confidence score, entity type, decision word.
- **Title** (semibold, 0.875rem, sans, uppercase, 0.15em tracking): page-level name only — Policy / Audit / Metrics.
- **Label** (medium, 0.75rem, sans, uppercase, 0.05–0.2em tracking): column headers, field labels, badge-surrounding chrome, the nav wordmark.
- **Body** (regular, 0.875rem, sans): descriptions, dry-run help text, form labels.

### Named Rules
**The Instrument/Chrome Split Rule.** If it is a fact the gateway produced (a number, a timestamp, an entity type, a decision), it renders in mono. If it is UI scaffolding the console itself authored (a label, a nav item, a button), it renders in sans. Mixing the two within one role is a drift signal. The rule is visible in the type scale itself, not just the font assignment: the tallest text in the system is a mono number, never a sans headline.

## Layout

Three console pages (`/policies`, `/audit`, `/metrics`) read as three channels on one recording desk behind a persistent, minimal channel strip (`NavBar`) — not a heavy left rail. The strip carries the recording-pulse kinetic tell. Page content sits in a centered `max-w-6xl` container (`px-4 py-6`), and each page opens with a `text-sm uppercase` title underlined by a 2px ink rule (`border-b-2 border-ink pb-2`).

Content lays out as tabular, gridded chart-row tables (`PolicyTable`, `AuditTable`) rather than card-soup — a policy or audit record is a row on a strip, not a floating card. Metric tiles use a `grid-cols-2 lg:grid-cols-4` responsive grid; two-panel layouts (policy form + dry-run, decision chart + latency chart) use `grid-cols-1 lg:grid-cols-2`. The single responsive breakpoint actually used in the build is `lg` (1024px) — no `sm`/`md`/`xl` variants appear.

Spacing rhythm runs on Tailwind's default 4px scale: page sections stack at `space-y-6` (24px), grids gap at `gap-4`–`gap-6` (16–24px), panel interior padding is `p-3`–`p-5` (12–20px), and table/row cells pad at `px-4 py-2` (16px/8px). Density stays high by design — whitespace never comes at the cost of legibility.

## Elevation & Depth

Flat by construction. Depth is conveyed through a three-tier line hierarchy, never a shadow: a light `hairline` (#ddd4bc) divides table rows; a heavier `hairline-strong` (#b8ac8c) borders every panel, input, and table container; a 2px full-`ink` rule marks structural emphasis (table header underline, nav bottom border, page-title underline). Interactive state is a wash, not a shadow: hover is a 3–5% ink tint (`hover:bg-ink/[0.03]`, `/[0.05]`); focus shifts a border from `hairline-strong` to full `ink` with `outline-none`.

Two exceptions, both implemented exactly as the seed specified:
1. **The audit detail drawer** (`AuditDetailDrawer`) is a true overlay — it uses Tailwind's `shadow-xl` (`0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1)`) over a `bg-ink/30` (dark: `bg-black/50`) scrim to separate it from the page beneath.
2. **The dry-run stamp landing** (`.stamp-land` in `index.css`): a 0.55s keyframe animation, delayed 0.8s after the trace draw finishes, that scales the stamp in from 1.6× and settles to 1×, passing through an explicit two-step overshoot (scale 0.92 at 55%, scale 1.04 at 80%) with a shadow that appears at the overshoot (`0 4px 8px rgba(0,0,0,0.22)` at 55%, fading to `0 2px 4px rgba(0,0,0,0.12)` at 80%, gone at 100%). The easing is `cubic-bezier(0.22, 1, 0.36, 1)` — an authored quint ease-out, not a bounce/elastic curve; the overshoot is hand-specified in the keyframe steps, not borrowed from a springy timing function. Both this and the `.recording-pulse` and `trace-draw` animations respect `prefers-reduced-motion` (the pulse holds at a fixed 0.75 opacity; the stamp and trace skip animation entirely — `RecorderStrip` gates this in JS via `matchMedia`, not CSS alone).

### Named Rules
**The Ink-Not-Shadow Rule.** If something needs to look more important, make its line heavier or its ink darker before reaching for a shadow. A shadow is reserved for the audit drawer's overlay and the stamp's landing instant, not for making a table row feel like a card.

## Shapes

Structural elements — tables, panels, chart frames, buttons, inputs — stay hard-edged and rectilinear (`rounded-none` is the implicit default; no `rounded-lg`/`rounded-xl` appears anywhere in the component set). The one deliberate exception is the decision stamp: `DecisionBadge` applies an asymmetric border-radius (`3px 7px 4px 8px`, matching the `--radius-stamp` token declared in `index.css`, applied via inline style rather than a generated utility class) *and* an asymmetric border-width (`2.5px 1.5px 2px 2px`) plus a small fixed per-decision rotation (`-rotate-1` allow, `rotate-1` mask, `-rotate-2` block) — a real stamp presses unevenly onto paper, and the rotation is fixed per decision (not randomized per render) so it reads as "this stamp," not jitter. This is the one hand-applied mark on an otherwise machine-precise surface. The only other rounded shapes in the system are functional, not decorative: `rounded-full` on the recording-pulse dot and native checkbox controls.

## Components

### Buttons
- **Shape:** hard rectangle, no radius (`rounded-none` default).
- **Primary** (`PolicyForm` submit, `DryRunPanel` "Test"): 2px `ink` border, `ink` fill, `paper` text, `px-4 py-1.5`, `text-sm font-medium`.
- **Hover:** fill drops to transparent, text becomes `ink` — an ink-outline inversion, not a shadow or color-accent hover.
- **Disabled:** `opacity-50`.
- **Secondary/text buttons** (Delete, Cancel, Clear filters, time-window presets): no border, `ink-muted` text that darkens to `ink` (or `block` for destructive) on hover; the Metrics time-window selector uses a solid `ink`-filled tab for the active state instead.

### Cards / Containers
- **Corner Style:** hard rectangle, no radius.
- **Background:** `paper`, or a faint `ink` wash (`bg-ink/[0.02]`–`[0.04]`) for recessed sub-panels (the recorder strip track, the dry-run result block).
- **Shadow Strategy:** none at rest — see Elevation & Depth. The one card-like container with a shadow is the audit detail drawer overlay.
- **Border:** `hairline-strong`, uniform 1px, on every panel/table/form container.
- **Internal Padding:** `p-3`–`p-5` depending on density.

### Inputs / Fields
- **Style:** 1px `hairline-strong` border, `paper` background, `rounded-none`, `px-3 py-1.5`, mono font for numeric/enum fields (entity type, action, min confidence), sans for free text (name, dry-run textarea).
- **Focus:** border shifts to full `ink`, native outline suppressed (`focus:outline-none focus:border-ink`) — no glow, no ring.
- **Error:** inline `block`-ink text below the field (`PolicyForm`'s name-required error); no red border treatment on the field itself.

### Navigation
- **Style:** a 2px `ink` bottom border on the header; a channel strip of three uppercase, tracked (`0.15em`) sans labels (Policy / Audit / Metrics), each an `NavLink` with its own 2px bottom border — `ink`/transparent depending on active state, `ink-muted` text fading to `ink` on hover.
- **Live indicator:** a 2×2 `rounded-full` dot in `ink` (never a severity color — see the Two-Ink Rule's reconciled clause), animated by `.recording-pulse` (opacity 0.35↔1 over 2.4s, ease-in-out, infinite).
- **Mobile treatment:** not built separately — the strip is a fixed flex row; no distinct mobile nav pattern exists in the current build.

### RecorderStrip (signature component)
A 400×60 SVG trace inside a hairline-bordered, faintly ink-washed track, overlaid with a 20-column calibration grid (10%-opacity `ink` verticals every 20 units) evoking a strip-chart's ruled paper. Each decision has one fixed path shape — a flat line for ALLOW, a moderate single bump for MASK, a sharp spike for BLOCK — stroked in that decision's ink. On completion the path animates with `trace-draw` (`stroke-dashoffset` from full length to 0, 0.9s ease-out), then the `DecisionBadge` lands with `.stamp-land` 0.8s later. While pending, the trace shows at 25% opacity under a pulsing "recording…" label.

### RedactedText (signature component)
Marks every `[REDACTED_*]` token in a redacted string in its severity ink — underlined (`decoration-2`), not boxed or background-highlighted — like a technician circling an anomaly on a chart. The category rule is exact, not heuristic: the token string `[REDACTED_SECRET]` is always BLOCK-ink (the backend collapses every secret entity type to that one literal token), and every other `[REDACTED_*]` token is always MASK-ink PII. Used everywhere redacted gateway text appears: dry-run result, audit table preview, audit detail drawer's prompt/response.

### MetricCard
A hairline-bordered instrument plate: uppercase muted label on top, a large (`text-2xl`) mono ink number as the headline value, an optional muted mono sublabel beneath. This is the one place the type ramp's ceiling (24px) appears, and it's data, not chrome.

### Tables (PolicyTable, AuditTable)
Hairline-bordered container; header row sits on a 2px `ink` rule with muted uppercase-tracked sans labels; body rows divide on plain `hairline`, hover with a 3–5% ink wash, and carry a `DecisionBadge` in the decision column. Mono for every data cell (entity type, timestamp, latency, confidence); sans only for the policy name. Empty states render via `EmptyState` (dashed hairline-strong border, centered text) rather than an empty table shell.

### Charts (DecisionTimeseriesChart, LatencyChart)
Recharts themed to match the instrument, not left at defaults: `hairline` for the Cartesian grid, `hairline-strong` for axis lines, mono ticks in `ink-muted`, a hard-edged (`borderRadius: 0`) tooltip with a 1px `ink` border. `DecisionTimeseriesChart` stacks three ink-colored areas (allow/mask/block, opacity 0.12/0.35/0.45) rather than introducing new chart colors. `LatencyChart` plots p95 only, in plain `ink` — latency is not a severity event, so it deliberately does not borrow `block`.

## Do's and Don'ts

### Do:
- **Do** keep the mono/sans split absolute: gateway-produced data in mono, console-authored chrome in sans, no exceptions.
- **Do** let the dry-run panel be the one place motion and a landing shadow concentrate — `trace-draw` + `.stamp-land`, both quint-eased, never bounce/elastic.
- **Do** treat the audit table and policy table as instrument-strip rows, dense and scannable, before treating them as anything decorative.
- **Do** make the two severity inks confident and saturated — restraint is their rarity of use, not their intensity when used.
- **Do** give the paper ground real two-layer chain-line texture and keep the one kinetic pulse in the nav shell, in graphite, never a severity color.
- **Do** respect `prefers-reduced-motion` on every animation (`.recording-pulse`, `.stamp-land`, `trace-draw`) — the built system gates all three, two in CSS and one in JS.

### Don't:
- **Don't** introduce a third accent color. Two severity inks, full stop.
- **Don't** default to rounded cards with soft shadows — no `rounded-lg`/`rounded-xl` appears anywhere in the built component set, and it should stay that way outside the one stamp exception.
- **Don't** reach for a dark, neon-accented "hacker terminal" palette. Dark mode is the same paper/ink pair under night-shift lighting, not a hue-shifted swap.
- **Don't** use a severity ink as a permanent at-rest UI accent (nav, brand mark, link). It may extend to failure/destructive states (errors, crash boundary, delete-confirm) because those are real severity-adjacent events — that is the one place the seed's original "never a UI accent" language was too strict for what the build correctly does.
- **Don't** reach for the unused `-tint` color pairs (`block-tint`, `mask-tint`) without a real second use case; they exist in `index.css` but nothing in the shipped UI reads them yet.
