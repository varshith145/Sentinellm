import { NavLink } from "react-router-dom";

const CHANNELS = [
  { to: "/policies", label: "Policy" },
  { to: "/audit", label: "Audit" },
  { to: "/metrics", label: "Metrics" },
];

export function NavBar() {
  return (
    <header className="border-b-2 border-ink bg-paper dark:border-ink-dark dark:bg-paper-dark">
      <div className="mx-auto flex max-w-6xl items-center gap-8 px-4 py-3">
        <div className="flex items-center gap-2">
          {/* Ink, not a severity color — this pulse is a permanent, at-rest
              UI element, and the Two-Ink Rule reserves vermillion/ochre for
              actual BLOCK/MASK events. Using a severity ink here would be
              exactly the drift the rule exists to catch. */}
          <span
            aria-hidden
            className="recording-pulse h-2 w-2 rounded-full bg-ink dark:bg-ink-dark"
            title="Live"
          />
          <span className="text-xs font-semibold tracking-[0.2em] text-ink uppercase dark:text-ink-dark">
            SentinelLM
          </span>
        </div>
        <nav className="flex gap-6">
          {CHANNELS.map((channel) => (
            <NavLink
              key={channel.to}
              to={channel.to}
              className={({ isActive }) =>
                `border-b-2 py-1 text-xs font-medium tracking-[0.15em] uppercase transition-colors ${
                  isActive
                    ? "border-ink text-ink dark:border-ink-dark dark:text-ink-dark"
                    : "border-transparent text-ink-muted hover:text-ink dark:text-ink-muted-dark dark:hover:text-ink-dark"
                }`
              }
            >
              {channel.label}
            </NavLink>
          ))}
        </nav>
      </div>
    </header>
  );
}
