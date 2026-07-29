import { Component, type ErrorInfo, type ReactNode } from "react";

interface Props {
  children: ReactNode;
}

interface State {
  error: Error | null;
}

/**
 * A React error boundary must be a class component — there is still no
 * hook-based equivalent (getDerivedStateFromError/componentDidCatch have
 * no hooks form as of React 19).
 */
export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("Console crashed:", error, info.componentStack);
  }

  render() {
    if (this.state.error) {
      return (
        <div className="m-6 border-2 border-block p-6 dark:border-block-dark">
          <p className="font-medium text-block dark:text-block-dark">
            The console hit an unexpected error.
          </p>
          <p className="mt-1 font-mono text-sm text-ink-muted dark:text-ink-muted-dark">
            {this.state.error.message}
          </p>
        </div>
      );
    }
    return this.props.children;
  }
}
