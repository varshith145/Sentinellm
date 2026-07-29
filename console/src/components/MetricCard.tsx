export function MetricCard({
  label,
  value,
  sublabel,
}: {
  label: string;
  value: string;
  sublabel?: string;
}) {
  return (
    <div className="border border-hairline-strong px-4 py-3 dark:border-hairline-strong-dark">
      <p className="text-xs font-medium tracking-wide text-ink-muted uppercase dark:text-ink-muted-dark">
        {label}
      </p>
      <p className="mt-1 font-mono text-2xl font-semibold text-ink dark:text-ink-dark">
        {value}
      </p>
      {sublabel && (
        <p className="mt-0.5 font-mono text-xs text-ink-muted dark:text-ink-muted-dark">
          {sublabel}
        </p>
      )}
    </div>
  );
}
