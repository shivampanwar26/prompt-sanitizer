import { useMemo, useState } from 'react'

const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'
const example = 'My email is shivam@gmail.com and I have a fever.'

async function request(path, body) {
  const response = await fetch(`${API_URL}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  const payload = await response.json().catch(() => ({}))
  if (!response.ok) throw new Error(payload.detail || 'The API request failed.')
  return payload
}

function downloadJsonl(rows) {
  const contents = rows.map((row) => JSON.stringify(row)).join('\n') + '\n'
  const blob = new Blob([contents], { type: 'application/x-ndjson' })
  const anchor = document.createElement('a')
  anchor.href = URL.createObjectURL(blob)
  anchor.download = 'prosan-training.jsonl'
  anchor.click()
  URL.revokeObjectURL(anchor.href)
}

export default function App() {
  const [tab, setTab] = useState('sanitize')
  const [prompt, setPrompt] = useState(example)
  const [mode, setMode] = useState('pii_only')
  const [result, setResult] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [batchText, setBatchText] = useState('')
  const [batchResult, setBatchResult] = useState([])

  const piiCount = useMemo(() => result?.selected_words?.length || 0, [result])

  async function sanitize() {
    if (!prompt.trim()) return
    setBusy(true); setError('')
    try {
      setResult(await request('/sanitize', { prompt, mode }))
    } catch (err) {
      setError(`Cannot reach ProSan at ${API_URL}: ${err.message}`)
    } finally {
      setBusy(false)
    }
  }

  async function createDataset() {
    const prompts = batchText.split(/\r?\n/).map((line) => line.trim()).filter(Boolean)
    if (!prompts.length) { setError('Add at least one prompt, with one prompt per line.'); return }
    setBusy(true); setError('')
    try {
      const rows = await request('/sanitize/batch', { prompts, mode })
      setBatchResult(rows)
      downloadJsonl(rows)
    } catch (err) {
      setError(`Dataset creation failed: ${err.message}`)
    } finally {
      setBusy(false)
    }
  }

  function importPrompts(file) {
    const reader = new FileReader()
    reader.onload = () => setBatchText(String(reader.result || ''))
    reader.readAsText(file)
  }

  return (
    <main className="shell">
      <header className="topbar">
        <div className="brand"><span className="brand-mark">P</span><span>ProSan</span></div>
        <span className="eyebrow">Prompt privacy workspace</span>
        <span className="api-pill">API {API_URL.replace(/^https?:\/\//, '')}</span>
      </header>

      <section className="hero">
        <p className="kicker">PRIVACY-PRESERVING LANGUAGE PIPELINE</p>
        <h1>Sanitize prompts.<br /><em>Keep the meaning.</em></h1>
        <p className="hero-copy">Create safe prompt pairs on a capable machine, then train and deploy a compact local sanitizer.</p>
      </section>

      <nav className="tabs" aria-label="Workspace sections">
        <button className={tab === 'sanitize' ? 'active' : ''} onClick={() => setTab('sanitize')}>01 · Sanitize</button>
        <button className={tab === 'dataset' ? 'active' : ''} onClick={() => setTab('dataset')}>02 · Dataset</button>
        <button className={tab === 'deploy' ? 'active' : ''} onClick={() => setTab('deploy')}>03 · Deploy locally</button>
      </nav>

      {error && <div className="alert" role="alert">{error}</div>}

      {tab === 'sanitize' && <section className="workspace two-column">
        <div className="card input-card">
          <div className="card-heading"><div><p className="step">INPUT PROMPT</p><h2>Find high-confidence PII</h2></div><span className="status">Ready</span></div>
          <textarea value={prompt} onChange={(event) => setPrompt(event.target.value)} aria-label="Prompt to sanitize" />
          <div className="controls">
            <label>Policy
              <select value={mode} onChange={(event) => setMode(event.target.value)}>
                <option value="pii_only">Safe PII redaction</option>
                <option value="prosan">Experimental ProSan</option>
              </select>
            </label>
            <button className="primary" onClick={sanitize} disabled={busy || !prompt.trim()}>{busy ? 'Sanitizing…' : 'Sanitize prompt →'}</button>
          </div>
          <p className="hint">Safe mode redacts emails, phones, cards, URLs, IP addresses, and API keys without rewriting medical content.</p>
        </div>

        <div className="card output-card">
          <div className="card-heading"><div><p className="step">SANITIZED PROMPT</p><h2>{result ? `${piiCount} protected span${piiCount === 1 ? '' : 's'}` : 'Awaiting input'}</h2></div><span className="lock">⌁</span></div>
          <div className={result ? 'result-text' : 'result-text empty'}>{result?.sanitized || 'Your privacy-preserving prompt will appear here.'}</div>
          <div className="spans">
            {result?.selected_words?.map((item) => <div className="span-row" key={`${item.start}-${item.end}`}><span>{item.pos_tag.replace('PII:', '')}</span><code>{item.word}</code><b>{item.replacement}</b></div>)}
            {result && !piiCount && <p>No high-confidence PII detected; the prompt is unchanged.</p>}
          </div>
        </div>
      </section>}

      {tab === 'dataset' && <section className="workspace dataset-layout">
        <div className="card">
          <p className="step">SANITIZED DATASET GENERATION</p><h2>Build training pairs</h2>
          <p className="body-copy">Paste raw prompts or import a text file. The browser downloads JSONL after the API produces original → sanitized pairs.</p>
          <div className="file-line"><label className="file-button">Import .txt<input type="file" accept=".txt,text/plain" onChange={(event) => event.target.files?.[0] && importPrompts(event.target.files[0])} /></label><span>One prompt per line</span></div>
          <textarea className="batch-area" value={batchText} onChange={(event) => setBatchText(event.target.value)} placeholder={'My email is a@b.com\nCall +91 98765 43210\nI have a fever'} />
          <div className="controls"><label>Policy
            <select value={mode} onChange={(event) => setMode(event.target.value)}><option value="pii_only">Safe PII redaction</option><option value="prosan">Experimental ProSan</option></select>
          </label><button className="primary" onClick={createDataset} disabled={busy}>{busy ? 'Generating…' : 'Generate & download JSONL'}</button></div>
        </div>
        <aside className="pipeline-card"><p className="step">PIPELINE</p><div className="pipeline"><span>Raw prompts</span><i>↓</i><span>Protected pairs</span><i>↓</i><span>training.jsonl</span></div><p>{batchResult.length ? `${batchResult.length} pairs generated in this session.` : 'Training pairs preserve unchanged prompts too, teaching the model when to leave text alone.'}</p></aside>
      </section>}

      {tab === 'deploy' && <section className="workspace deploy-grid">
        <div className="card command-card"><p className="step">FINE-TUNING</p><h2>Train a compact local model</h2><code>python -m app.cli --train data/training.jsonl --output-dir models/local-sanitizer</code><p className="body-copy">Uses Flan-T5 Small by default. Run this on a machine with enough memory and network access for the initial model download.</p></div>
        <div className="card command-card"><p className="step">LOW-RESOURCE INFERENCE</p><h2>Deploy locally</h2><code>python -m app.cli --local-model models/local-sanitizer "Email me@example.com"</code><p className="body-copy">The saved model runs from the local directory and no hosted LLM service is required for inference.</p></div>
      </section>}
    </main>
  )
}
