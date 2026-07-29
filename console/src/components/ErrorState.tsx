export function ErrorState({ message }: { message: string }) {
  return (
    <div className="flex flex-col items-center justify-center border-2 border-block px-6 py-16 text-center dark:border-block-dark">
      <p className="text-sm font-medium text-block dark:text-block-dark">
        Something went wrong
      </p>
      <p className="mt-1 max-w-sm font-mono text-sm text-ink-muted dark:text-ink-muted-dark">
        {message}
      </p>
    </div>
  );
}
