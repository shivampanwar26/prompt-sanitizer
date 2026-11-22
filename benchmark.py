"""Benchmark the sanitizer on a labelled prompt set, optionally against a baseline.

Each row of data/benchmark.jsonl has a prompt, the gold ``sensitive`` strings
that must disappear, and ``keep`` strings the task depends on.

Metrics
  PHR              % of gold sensitive items absent from the output (higher = more private)
  Utility          % of task-critical ``keep`` items still present (higher = more useful)
  Collateral       % of non-sensitive words that were changed anyway (lower = better)
  Benign intact    % of prompts without PII returned unchanged (higher = better)
  PPL ratio        sanitized / original perplexity under distilgpt2 (closer to 1 = more fluent)
  Latency          mean seconds per prompt, models warm, evaluation disabled

Usage
  python benchmark.py                                  # this checkout only, CUDA when available
  python benchmark.py --baseline ../prompt-sanitizer-old   # compare with another checkout
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# Windows consoles default to cp1252; benchmark prompts contain characters
# (e.g. "°") that encoding cannot represent, which would otherwise crash --show.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "benchmark.jsonl"


def _contains(text, item):
    pattern = (r"(?<!\w)" if item[:1].isalnum() else "") + re.escape(item) + (r"(?!\w)" if item[-1:].isalnum() else "")
    return re.search(pattern, text, re.I) is not None


def run_worker(root: str, device: str) -> None:
    """Sanitize every benchmark prompt with the package at ``root``; print JSON."""
    os.chdir(root)
    sys.path.insert(0, root)
    from sanitizer import PromptSanitizer

    sanitizer = PromptSanitizer(device=device)
    rows = [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines() if line.strip()]

    sanitizer.sanitize("Warm up the models, John.", evaluate=False)
    outputs = []
    for row in rows:
        t0 = time.perf_counter()
        result = sanitizer.sanitize(row["prompt"], evaluate=False)
        outputs.append({
            "text": result.text,
            "latency": time.perf_counter() - t0,
            "edits": [{"start": e["start"], "end": e["end"]} for e in result.selected_words],
        })
    print("@@RESULT@@" + json.dumps(outputs))


def collect(root: str, device: str):
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker", root, "--device", device],
        capture_output=True, text=True, encoding="utf-8",
    )
    marker = [line for line in proc.stdout.splitlines() if line.startswith("@@RESULT@@")]
    if proc.returncode != 0 or not marker:
        sys.stderr.write(proc.stderr[-4000:])
        raise SystemExit(f"benchmark worker failed for {root}")
    return json.loads(marker[0][len("@@RESULT@@"):])


def score(rows, outputs, evaluator):
    sens_total = sens_hidden = keep_total = keep_kept = 0
    collat_total = collat_changed = 0
    benign = benign_intact = 0
    token_tp = token_fp = token_fn = token_tn = 0
    ratios = []
    word_re = re.compile(r"\w+(?:['’.@+\-]\w+)*")

    for row, out in zip(rows, outputs):
        prompt, text = row["prompt"], out["text"]
        for item in row["sensitive"]:
            sens_total += 1
            sens_hidden += not _contains(text, item)
        gold_spans = [
            (m.start(), m.end())
            for item in row["sensitive"]
            for m in re.finditer(re.escape(item), prompt, flags=re.I)
        ]
        predicted_spans = [(e["start"], e["end"]) for e in out.get("edits", [])]
        # Token-level PII detection metrics. Accuracy is paired with P/R/F1
        # because non-sensitive tokens are the majority class.
        for token in word_re.finditer(prompt):
            gold_positive = any(token.start() < end and token.end() > start for start, end in gold_spans)
            predicted_positive = any(token.start() < end and token.end() > start for start, end in predicted_spans)
            if gold_positive and predicted_positive:
                token_tp += 1
            elif predicted_positive:
                token_fp += 1
            elif gold_positive:
                token_fn += 1
            else:
                token_tn += 1
        for item in row["keep"]:
            keep_total += 1
            keep_kept += _contains(text, item)

        # Words outside every gold sensitive span that did not survive.
        covered = [(m.start(), m.end()) for s in row["sensitive"] for m in re.finditer(re.escape(s), prompt)]
        for m in word_re.finditer(prompt):
            if any(m.start() < e and m.end() > s for s, e in covered):
                continue
            collat_total += 1
            collat_changed += not _contains(text, m.group(0))

        if not row["sensitive"]:
            benign += 1
            benign_intact += text == prompt

        orig_ppl, san_ppl = evaluator.perplexity_batch([prompt, text])
        if orig_ppl > 0:
            ratios.append(san_ppl / orig_ppl)

    token_precision = token_tp / (token_tp + token_fp) if token_tp + token_fp else 0.0
    token_recall = token_tp / (token_tp + token_fn) if token_tp + token_fn else 0.0
    token_f1 = 2 * token_precision * token_recall / (token_precision + token_recall) if token_precision + token_recall else 0.0
    token_total = token_tp + token_fp + token_fn + token_tn
    return {
        "PHR": 100.0 * sens_hidden / max(sens_total, 1),
        "Utility": 100.0 * keep_kept / max(keep_total, 1),
        "Token accuracy": 100.0 * (token_tp + token_tn) / max(token_total, 1),
        "Token precision": 100.0 * token_precision,
        "Token recall": 100.0 * token_recall,
        "Token F1": 100.0 * token_f1,
        "Collateral": 100.0 * collat_changed / max(collat_total, 1),
        "Benign intact": 100.0 * benign_intact / max(benign, 1),
        "PPL ratio": sum(ratios) / max(len(ratios), 1),
        "Latency (s)": sum(o["latency"] for o in outputs) / max(len(outputs), 1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", help="Path to another checkout to compare against")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--show", action="store_true", help="Print every sanitized prompt")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        run_worker(args.worker, args.device)
        return

    sys.path.insert(0, str(HERE))
    from sanitizer.evaluator import PromptEvaluator
    from sanitizer.model_registry import resolve_device

    device = str(resolve_device(args.device))

    rows = [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines() if line.strip()]
    systems = {}
    if args.baseline:
        systems["Baseline"] = collect(str(Path(args.baseline).resolve()), device)
    systems["ProSan+"] = collect(str(HERE), device)

    evaluator = PromptEvaluator(model_name="distilgpt2", device=device)
    scores = {name: score(rows, outs, evaluator) for name, outs in systems.items()}

    if args.show:
        for i, row in enumerate(rows):
            print(f"\n[{row['category']}] {row['prompt']}")
            for name, outs in systems.items():
                print(f"  {name:<9} {outs[i]['text']}")

    names = list(scores)
    print(f"\n{len(rows)} prompts, device={device}\n")
    print(f"{'Metric':<16}" + "".join(f"{n:>12}" for n in names))
    print("-" * (16 + 12 * len(names)))
    for metric in scores[names[0]]:
        fmt = "{:>12.3f}" if metric in ("PPL ratio", "Latency (s)") else "{:>11.1f}%"
        print(f"{metric:<16}" + "".join(fmt.format(scores[n][metric]) for n in names))


if __name__ == "__main__":
    main()
