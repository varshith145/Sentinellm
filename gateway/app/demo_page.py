"""
Static HTML for the public demo landing page served at ``GET /``.

Vanilla HTML/CSS/JS — no build step, no external dependencies. It calls the
``POST /scan`` endpoint and renders the decision, findings, redacted text, and
latency so a non-technical visitor can test the detection pipeline in seconds.
"""

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>SentinelLM — Live PII / Secret Detection Demo</title>
<style>
  :root {
    --bg: #0f1420; --card: #182030; --line: #2a3346; --text: #e7ecf3;
    --muted: #94a3b8; --accent: #6366f1;
    --allow: #16a34a; --mask: #d97706; --block: #dc2626;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--text);
    font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }
  .wrap { max-width: 860px; margin: 0 auto; padding: 32px 20px 64px; }
  header h1 { margin: 0 0 4px; font-size: 26px; letter-spacing: -0.01em; }
  header p { margin: 0; color: var(--muted); }
  .badge {
    display: inline-block; margin-top: 10px; padding: 4px 10px; border-radius: 999px;
    background: #1e293b; color: var(--muted); font-size: 12px; border: 1px solid var(--line);
  }
  .card {
    background: var(--card); border: 1px solid var(--line); border-radius: 12px;
    padding: 18px; margin-top: 20px;
  }
  label { font-weight: 600; font-size: 13px; color: var(--muted); display: block; margin-bottom: 8px; }
  textarea {
    width: 100%; min-height: 110px; resize: vertical; padding: 12px; border-radius: 8px;
    border: 1px solid var(--line); background: #0d121d; color: var(--text);
    font: 14px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace;
  }
  .examples { display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0 4px; }
  .examples button {
    background: #1e2536; color: var(--text); border: 1px solid var(--line);
    border-radius: 8px; padding: 7px 12px; font-size: 13px; cursor: pointer;
  }
  .examples button:hover { border-color: var(--accent); }
  .scan-btn {
    margin-top: 14px; background: var(--accent); color: #fff; border: 0;
    border-radius: 8px; padding: 11px 22px; font-size: 15px; font-weight: 600; cursor: pointer;
  }
  .scan-btn:disabled { opacity: 0.5; cursor: default; }
  #result { display: none; }
  .decision {
    display: inline-block; padding: 6px 14px; border-radius: 8px; font-weight: 700;
    letter-spacing: 0.04em; color: #fff;
  }
  .decision.ALLOW { background: var(--allow); }
  .decision.MASK  { background: var(--mask); }
  .decision.BLOCK { background: var(--block); }
  .meta { color: var(--muted); font-size: 13px; margin-left: 12px; }
  table { width: 100%; border-collapse: collapse; margin-top: 14px; font-size: 13px; }
  th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--line); }
  th { color: var(--muted); font-weight: 600; }
  td code { background: #0d121d; padding: 2px 6px; border-radius: 5px; }
  .tag { font-size: 11px; padding: 2px 7px; border-radius: 999px; background: #1e293b; border: 1px solid var(--line); }
  .redacted {
    margin-top: 14px; padding: 12px; border-radius: 8px; background: #0d121d;
    border: 1px solid var(--line); font: 14px/1.5 ui-monospace, monospace; white-space: pre-wrap;
  }
  .note { color: var(--muted); font-size: 12px; margin-top: 18px; }
  a { color: #818cf8; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>🛡️ SentinelLM</h1>
    <p>AI security gateway — a three-pass pipeline (regex → Presidio → fine-tuned DistilBERT) that detects PII &amp; secrets in text.</p>
    <span class="badge">Demo mode — detection pipeline only; LLM proxy disabled</span>
  </header>

  <div class="card">
    <label for="text">Text to scan</label>
    <textarea id="text" placeholder="Type or paste text, or pick an example below…"></textarea>
    <div class="examples">
      <button data-ex="Email me at john.smith@company.com or call 415-555-0132.">Plain PII (regex)</button>
      <button data-ex="reach me at john dot smith at company dot com">Obfuscated PII (semantic)</button>
      <button data-ex="Here is the key: AKIAIOSFODNN7EXAMPLE">AWS secret (block)</button>
      <button data-ex="We use AWS for cloud hosting and it scales well.">Benign (allow)</button>
    </div>
    <button class="scan-btn" id="scanBtn">Scan</button>
  </div>

  <div class="card" id="result">
    <div>
      <span class="decision" id="decision"></span>
      <span class="meta" id="meta"></span>
    </div>
    <div id="findingsWrap"></div>
    <div id="redactedWrap"></div>
  </div>

  <p class="note">
    Detection runs on CPU. First load after inactivity may take ~30s while the Space wakes.
    Full API docs at <a href="/docs">/docs</a> · raw endpoint <code>POST /scan</code>.
  </p>
</div>

<script>
const $ = (id) => document.getElementById(id);
document.querySelectorAll('.examples button').forEach(b => {
  b.addEventListener('click', () => { $('text').value = b.dataset.ex; });
});

function escapeHtml(s) {
  return (s ?? '').replace(/[&<>"']/g, c => (
    {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]
  ));
}

async function scan() {
  const text = $('text').value.trim();
  if (!text) return;
  const btn = $('scanBtn');
  btn.disabled = true; btn.textContent = 'Scanning…';
  try {
    const res = await fetch('/scan', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });
    const data = await res.json();
    render(data);
  } catch (e) {
    $('result').style.display = 'block';
    $('decision').className = 'decision BLOCK';
    $('decision').textContent = 'ERROR';
    $('meta').textContent = String(e);
    $('findingsWrap').innerHTML = '';
    $('redactedWrap').innerHTML = '';
  } finally {
    btn.disabled = false; btn.textContent = 'Scan';
  }
}

function render(data) {
  $('result').style.display = 'block';
  $('decision').className = 'decision ' + data.decision;
  $('decision').textContent = data.decision;
  const dets = (data.detectors_used || []).join(', ');
  $('meta').textContent = `${data.findings.length} finding(s) · ${data.latency_ms} ms · detectors: ${dets}`;

  if (data.findings.length) {
    let rows = data.findings.map(f => `
      <tr>
        <td><span class="tag">${escapeHtml(f.entity_type)}</span></td>
        <td><code>${escapeHtml(f.text)}</code></td>
        <td>${(f.confidence * 100).toFixed(1)}%</td>
        <td>${escapeHtml(f.detector)}</td>
      </tr>`).join('');
    $('findingsWrap').innerHTML = `
      <table>
        <thead><tr><th>Entity</th><th>Matched text</th><th>Confidence</th><th>Caught by</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>`;
  } else {
    $('findingsWrap').innerHTML = '<p class="note">No sensitive entities detected.</p>';
  }

  if (data.decision !== 'ALLOW') {
    $('redactedWrap').innerHTML =
      `<label style="margin-top:16px;">Redacted output</label>
       <div class="redacted">${escapeHtml(data.redacted_text)}</div>`;
  } else {
    $('redactedWrap').innerHTML = '';
  }
}

$('scanBtn').addEventListener('click', scan);
$('text').addEventListener('keydown', (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') scan();
});
</script>
</body>
</html>
"""
