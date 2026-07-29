/**
 * DIRECTION CONTRACT — "The Instrument Room" (seed key e9e4f728, assigned
 * index 4 of 7, concept-seed --scope direction --mode operate)
 *
 * THESIS: a security console instrumented like a live seismograph, not
 * skinned like a SaaS dashboard — refuses both the indigo-and-slate
 * default this project already had and the "hacker terminal" cliché.
 *
 * OWN-WORLD: warm paper ground with real grain, black ink structure,
 * exactly two severity inks (confident vermillion/BLOCK, ochre/MASK,
 * never a third), IBM Plex Mono for every gateway-produced value, system
 * sans for chrome, flat-by-construction — ink weight and hairline rule
 * over shadow, except the drawer overlay and the dry-run stamp landing.
 *
 * STORY: a visitor reads dense, real policy/audit/metrics data at a
 * glance, edits a rule and watches a real decision flip live, and tests
 * text against a rule by watching a trace draw and land on a stamped
 * verdict.
 *
 * FIRST VIEWPORT: a channel strip (Policy / Audit / Metrics) carrying a
 * live recording pulse; below it, a hairline chart-row table with
 * ink-stamped decision marks, never rounded-card soup.
 *
 * FORM: extend-an-established-surface build — composition (three pages,
 * their tasks and REST contract) was already working and fixed; only the
 * visual world changed. No surface-level concept roll applied; the
 * world-selection roll happened at the direction stage above.
 *
 * See DESIGN.md (repo root) and .impeccable/surfaces/console.md for the
 * full system this contract commits to.
 */
import { Navigate, Route, Routes } from "react-router-dom";

import { ErrorBoundary } from "@/components/ErrorBoundary";
import { NavBar } from "@/components/NavBar";
import { AuditPage } from "@/routes/AuditPage";
import { MetricsPage } from "@/routes/MetricsPage";
import { PoliciesPage } from "@/routes/PoliciesPage";

function App() {
  return (
    <div className="min-h-screen">
      <NavBar />
      <main className="mx-auto max-w-6xl px-4 py-6">
        <ErrorBoundary>
          <Routes>
            <Route path="/" element={<Navigate to="/policies" replace />} />
            <Route path="/policies" element={<PoliciesPage />} />
            <Route path="/audit" element={<AuditPage />} />
            <Route path="/metrics" element={<MetricsPage />} />
          </Routes>
        </ErrorBoundary>
      </main>
    </div>
  );
}

export default App;
