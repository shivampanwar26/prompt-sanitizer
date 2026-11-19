# ProSan — Utility-Based Prompt Privacy Sanitizer

A clean, unified implementation of the ProSan prompt-sanitization pipeline described in *"ProSan: Utility-Based Prompt Privacy Sanitizer"* (IEEE TIFS 2026).

## Pipeline Workflow

```
Original Prompt
  └─► PII Pattern Redaction (emails, phone numbers, IPs, API keys, card numbers)
  └─► Gradient Utility Importance Score (K_w)
  └─► Privacy Self-Information Score (O_w) & Repeated Entity Max Pooling
  └─► Adaptive Word Selection & POS Filtering (γ_q)
  └─► Masked LM Candidate Prediction & Cumulative Truncation (τ)
  └─► WordNet Path Similarity (s_i) & Softmax Probability Recalculation (p'_i)
  └─► Sanitized Prompt Output
```

## Quick Start & Requirements

Python 3.10+ recommended.

```bash
# Setup Virtual Environment
python -m venv .venv

# Activate:
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Default models loaded on first run:
- Importance & Privacy Self-Information: `distilgpt2`
- Candidate Replacement Generator: `roberta-base`

---

## Running the CLI

Run direct prompt desensitization via the clean CLI:

```bash
python -m app.cli "My name is John Doe and my email is john@example.com. I live in Delhi and have severe chest pain."
```

Or pass a file containing prompts (one per line):

```bash
python -m app.cli -f prompts.txt
```

Or evaluate readability perplexity ($PPL$):

```bash
python -m app.cli --eval "My name is John Doe and I live in Delhi."
```

---

## Running the REST API

```bash
uvicorn app.main:app --reload
```

Send a POST request to `/sanitize`:

```json
{
  "prompt": "My name is John Doe and my email is john@example.com."
}
```

---

## Web Dashboard (React/Vite)

Start the API and frontend in separate terminals:

```bash
# Terminal 1: API
uvicorn app.main:app --port 8000

# Terminal 2: Web UI
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` in your web browser.
