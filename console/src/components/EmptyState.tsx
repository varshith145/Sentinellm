export function EmptyState({
  title,
  description,
}: {
  title: string;
  description?: string;
}) {
  return (
    <div className="flex flex-col items-center justify-center border border-dashed border-hairline-strong px-6 py-16 text-center dark:border-hairline-strong-dark">
      <p className="text-sm font-medium text-ink dark:text-ink-dark">{title}</p>
      {description && (
        <p className="mt-1 max-w-sm text-sm text-ink-muted dark:text-ink-muted-dark">
          {description}
        </p>
      )}
    </div>
  );
}
