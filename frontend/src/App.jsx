import { useEffect, useMemo, useState } from 'react'

const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'
const example = 'My email is shivam@gmail.com and I have a fever.'
const ACCEPTED_IMAGES = 'image/png,image/jpeg,image/webp,image/bmp,image/tiff'

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

async function sanitizeImageRequest(file, options) {
  const form = new FormData()
  form.append('file', file)
  const query = new URLSearchParams({
    redact_faces: String(options.redactFaces),
    redact_text: String(options.redactText),
    deep: String(options.deep),
  })
  const response = await fetch(`${API_URL}/sanitize/image?${query}`, { method: 'POST', body: form })
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}))
    throw new Error(payload.detail || 'The API request failed.')
  }
  let report = null
  try {
    report = JSON.parse(response.headers.get('X-Sanitization-Report') || 'null')
  } catch {
    report = null
  }
  return { blob: await response.blob(), report }
}

function sanitizedName(name) {
  const stem = name.replace(/\.[^.]+$/, '') || 'image'
  return `${stem}.sanitized.png`
}

function TextSanitizer({ setError }) {
  const [prompt, setPrompt] = useState(example)
  const [mode, setMode] = useState('pii_only')
  const [result, setResult] = useState(null)
  const [busy, setBusy] = useState(false)
  const [copied, setCopied] = useState(false)

  const piiCount = useMemo(() => result?.selected_words?.length || 0, [result])

  async function sanitize() {
    if (!prompt.trim()) return
    setBusy(true); setError(''); setCopied(false)
    try {
      setResult(await request('/sanitize', { prompt, mode }))
    } catch (err) {
      setError(`Cannot reach ProSan at ${API_URL}: ${err.message}`)
    } finally {
      setBusy(false)
    }
  }

  async function copyResult() {
    if (!result?.sanitized) return
    try {
      await navigator.clipboard.writeText(result.sanitized)
      setCopied(true)
    } catch {
      setError('Copying to the clipboard is not allowed in this browser.')
    }
  }

  return (
    <section className="workspace two-column">
      <div className="card input-card">
        <div className="card-heading"><div><p className="step">INPUT PROMPT</p><h2>Find sensitive information</h2></div><span className="status">Text</span></div>
        <textarea value={prompt} onChange={(event) => setPrompt(event.target.value)} aria-label="Prompt to sanitize" />
        <div className="controls">
          <label>Policy
            <select value={mode} onChange={(event) => setMode(event.target.value)}>
              <option value="pii_only">Safe PII redaction</option>
              <option value="prosan">ProSan+ (contextual)</option>
            </select>
          </label>
          <button className="primary" onClick={sanitize} disabled={busy || !prompt.trim()}>{busy ? 'Sanitizing…' : 'Sanitize prompt →'}</button>
        </div>
        <p className="hint">{mode === 'pii_only'
          ? 'Safe mode replaces names, addresses, emails, phones, cards, IDs, IPs and secrets with realistic surrogates, using rules and NER only.'
          : 'ProSan+ also scores every remaining word with a local language model and replaces rare identifying words that the task does not need.'}</p>
      </div>

      <div className="card output-card">
        <div className="card-heading">
          <div><p className="step">SANITIZED PROMPT</p><h2>{result ? `${piiCount} protected span${piiCount === 1 ? '' : 's'}` : 'Awaiting input'}</h2></div>
          {result ? <button className="ghost" onClick={copyResult}>{copied ? 'Copied' : 'Copy'}</button> : <span className="lock">⌁</span>}
        </div>
        <div className={result ? 'result-text' : 'result-text empty'}>{result?.sanitized || 'Your privacy-preserving prompt will appear here.'}</div>
        <div className="spans">
          {result?.selected_words?.map((item) => <div className="span-row" key={`${item.start}-${item.end}`}><span>{item.pos_tag.replace('PII:', '')}</span><code>{item.word}</code><b>{item.replacement}</b></div>)}
          {result && !piiCount && <p>No sensitive information detected; the prompt is unchanged.</p>}
        </div>
      </div>
    </section>
  )
}

function ImageSanitizer({ setError }) {
  const [file, setFile] = useState(null)
  const [originalUrl, setOriginalUrl] = useState('')
  const [sanitizedUrl, setSanitizedUrl] = useState('')
  const [report, setReport] = useState(null)
  const [busy, setBusy] = useState(false)
  const [dragging, setDragging] = useState(false)
  const [options, setOptions] = useState({ redactFaces: true, redactText: true, deep: false })

  useEffect(() => () => { if (originalUrl) URL.revokeObjectURL(originalUrl) }, [originalUrl])
  useEffect(() => () => { if (sanitizedUrl) URL.revokeObjectURL(sanitizedUrl) }, [sanitizedUrl])

  function chooseFile(next) {
    if (!next) return
    if (!next.type.startsWith('image/')) { setError('Choose an image file (PNG, JPEG, WebP, BMP or TIFF).'); return }
    setError('')
    setFile(next)
    setOriginalUrl(URL.createObjectURL(next))
    setSanitizedUrl('')
    setReport(null)
  }

  function onDrop(event) {
    event.preventDefault()
    setDragging(false)
    chooseFile(event.dataTransfer.files?.[0])
  }

  async function sanitize() {
    if (!file) return
    setBusy(true); setError('')
    try {
      const { blob, report: nextReport } = await sanitizeImageRequest(file, options)
      setSanitizedUrl(URL.createObjectURL(blob))
      setReport(nextReport)
    } catch (err) {
      setError(`Cannot reach ProSan at ${API_URL}: ${err.message}`)
    } finally {
      setBusy(false)
    }
  }

  const toggle = (key) => setOptions((current) => ({ ...current, [key]: !current[key] }))
  const nothingSelected = !options.redactFaces && !options.redactText

  return (
    <section className="workspace image-layout">
      <div className="card">
        <div className="card-heading"><div><p className="step">INPUT IMAGE</p><h2>Redact faces, text and metadata</h2></div><span className="status">Image</span></div>
        <label
          className={`dropzone${dragging ? ' dragging' : ''}${originalUrl ? ' filled' : ''}`}
          onDragOver={(event) => { event.preventDefault(); setDragging(true) }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
        >
          <input type="file" accept={ACCEPTED_IMAGES} onChange={(event) => chooseFile(event.target.files?.[0])} />
          {originalUrl
            ? <img src={originalUrl} alt="Original upload" />
            : <span><b>Drop an image here</b> or click to choose one<br /><small>PNG, JPEG, WebP, BMP or TIFF</small></span>}
        </label>
        {file && <p className="file-name">{file.name} · {(file.size / 1024).toFixed(0)} KB</p>}
        <div className="checks">
          <label><input type="checkbox" checked={options.redactFaces} onChange={() => toggle('redactFaces')} /> Pixelate faces</label>
          <label><input type="checkbox" checked={options.redactText} onChange={() => toggle('redactText')} /> Redact sensitive text (OCR)</label>
          <label className={options.redactText ? '' : 'disabled'}><input type="checkbox" checked={options.deep} disabled={!options.redactText} onChange={() => toggle('deep')} /> Deep text analysis (ProSan+)</label>
        </div>
        <div className="controls">
          <p className="hint">Metadata (EXIF, GPS, camera details) is always removed.</p>
          <button className="primary" onClick={sanitize} disabled={busy || !file}>{busy ? 'Sanitizing…' : 'Sanitize image →'}</button>
        </div>
        {nothingSelected && file && <p className="hint">Only metadata will be removed; turn on faces or text to redact content.</p>}
      </div>

      <div className="card output-card">
        <div className="card-heading">
          <div><p className="step">SANITIZED IMAGE</p><h2>{report ? `${report.faces_redacted + report.text_regions_redacted} region${report.faces_redacted + report.text_regions_redacted === 1 ? '' : 's'} redacted` : 'Awaiting input'}</h2></div>
          {sanitizedUrl ? <a className="ghost" href={sanitizedUrl} download={sanitizedName(file?.name || 'image')}>Download</a> : <span className="lock">⌁</span>}
        </div>
        <div className={sanitizedUrl ? 'image-result' : 'image-result empty'}>
          {sanitizedUrl ? <img src={sanitizedUrl} alt="Sanitized result" /> : <span>The sanitized image will appear here.</span>}
        </div>
        {report && <div className="report">
          <div className="stat"><b>{report.faces_redacted}</b><span>faces</span></div>
          <div className="stat"><b>{report.text_regions_redacted}</b><span>text lines</span></div>
          <div className="stat"><b>{report.metadata_removed.length}</b><span>metadata fields</span></div>
          {report.pii_kinds_found.length > 0 && <div className="report-row"><span>Found</span><div>{report.pii_kinds_found.map((kind) => <code key={kind}>{kind}</code>)}</div></div>}
          {report.metadata_removed.length > 0 && <div className="report-row"><span>Removed</span><div>{report.metadata_removed.map((kind) => <code key={kind}>{kind}</code>)}</div></div>}
          {report.warnings.map((warning) => <p className="warning" key={warning}>{warning}</p>)}
        </div>}
        {sanitizedUrl && !report && <p className="hint">The API did not return a report for this image.</p>}
      </div>
    </section>
  )
}

export default function App() {
  const [tab, setTab] = useState('text')
  const [error, setError] = useState('')

  function switchTab(next) { setTab(next); setError('') }

  return (
    <main className="shell">
      <header className="topbar">
        <div className="brand"><span className="brand-mark">P</span><span>ProSan+</span></div>
        <span className="eyebrow">Prompt privacy workspace</span>
        <span className="api-pill">API {API_URL.replace(/^https?:\/\//, '')}</span>
      </header>

      <section className="hero">
        <p className="kicker">PRIVACY-PRESERVING PROMPT SANITIZER</p>
        <h1>Sanitize prompts.<br /><em>Keep the meaning.</em></h1>
        <p className="hero-copy">Remove personal and secret information from text prompts and images before they reach a cloud LLM, while keeping what the task needs.</p>
      </section>

      <nav className="tabs" aria-label="Sanitizer type">
        <button className={tab === 'text' ? 'active' : ''} onClick={() => switchTab('text')}>01 · Text</button>
        <button className={tab === 'image' ? 'active' : ''} onClick={() => switchTab('image')}>02 · Image</button>
      </nav>

      {error && <div className="alert" role="alert">{error}</div>}

      {/* Both panels stay mounted so switching tabs keeps each one's input and result. */}
      <div hidden={tab !== 'text'}><TextSanitizer setError={setError} /></div>
      <div hidden={tab !== 'image'}><ImageSanitizer setError={setError} /></div>
    </main>
  )
}
