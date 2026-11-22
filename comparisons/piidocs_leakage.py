"""Measure ProSan+ with Rap-LI's PII extraction leakage metric on PIIDocs.

Rap-LI (Liu et al., IEEE TIFS 2026, Def. 6) defines the PII leakage rate of a
sanitized prompt as |I_orig ∩ I_san| / |I_orig|, where I_orig = E(S) are the PII
tokens extracted from the original and I_san = E(S~) those extracted from the
sanitized text, averaged over prompts. Here E(S) is the dataset's *gold* entity
list (not an NER model, so the sanitizer's own detector never grades itself).

Two readings are reported:
  exact   Rap-LI's definition: a gold PII token leaks if the identical token
          appears in the sanitized text (set intersection).
  strict  additionally, a token with digits or symbols leaks if half of it
          survives contiguously (e.g. a birth year kept by a date shift).
An e-mail address is split into its local part and domain; the reserved
documentation domains (example.com/.org/.net, RFC 2606), which identify no
one, are not counted as PII tokens.

PIIDocs is gretelai/gretel-pii-masking-en-v1 on the Hugging Face Hub (Rap-LI
used 4,119 documents of its train split; this script samples the test split).

    python comparisons/piidocs_leakage.py --parquet test-00000-of-00001.parquet --out results.json [--n 500] [--dump outputs.jsonl]
    python comparisons/piidocs_leakage.py --rescore outputs.jsonl --out results.json
"""

import argparse
import ast
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

STOP = {"a", "an", "the", "of", "and", "or", "in", "on", "at", "to", "for", "by", "with", "de", "la", "st", "rd", "nd", "th",
        "example.com", "example.org", "example.net"}
_EDGE = ".,;:!?-/'\"()"


def tokens(text):
    # Strip sentence punctuation from both ends so "2026." matches "2026" (applied to gold and
    # output alike); split e-mail addresses into local part and domain.
    out = []
    for t in re.findall(r"[\w@#$%&*!+.\-/]+", text.lower()):
        t = t.strip(_EDGE)
        parts = t.split("@", 1) if re.fullmatch(r"[^@\s]+@[^@\s]+\.[a-z]{2,}", t) else [t]
        out.extend(p for p in (p.strip(_EDGE) for p in parts) if len(p) >= 2)
    return out


def pii_tokens(entities):
    return {t for ent in entities for t in tokens(ent["entity"]) if t not in STOP}


def survives(token, out_tokens, out_lower, strict):
    if token in out_tokens:
        return True
    if strict and len(token) >= 6 and not token.isalpha():
        k = (len(token) + 1) // 2
        return any(token[i:i + k] in out_lower for i in range(len(token) - k + 1))
    return False


def score(records):
    """records: dicts with 'entities' and 'output'. Returns the metrics dict."""
    result = {"n_documents": len(records)}
    for mode, strict in (("exact", False), ("strict", True)):
        per_doc, by_type = [], defaultdict(lambda: [0, 0])
        tok_total = tok_leaked = 0
        for r in records:
            out = r["output"]
            out_tokens, out_lower = set(tokens(out)), out.lower()
            gold = pii_tokens(r["entities"])
            if gold:
                leaked = {t for t in gold if survives(t, out_tokens, out_lower, strict)}
                per_doc.append(len(leaked) / len(gold))
                tok_total += len(gold)
                tok_leaked += len(leaked)
            for ent in r["entities"]:
                kind = ent["types"][0] if ent.get("types") else "unknown"
                ent_tokens = [t for t in tokens(ent["entity"]) if t not in STOP]
                by_type[kind][0] += 1
                by_type[kind][1] += not any(survives(t, out_tokens, out_lower, strict) for t in ent_tokens)
        result[mode] = {
            "pii_leakage_rate_mean": 100 * sum(per_doc) / max(1, len(per_doc)),
            "pii_leakage_rate_micro": 100 * tok_leaked / max(1, tok_total),
            "entities_hidden": 100 * sum(v[1] for v in by_type.values()) / max(1, sum(v[0] for v in by_type.values())),
            "by_type": {k: {"n": v[0], "hidden_pct": 100 * v[1] / v[0]}
                        for k, v in sorted(by_type.items(), key=lambda kv: -kv[1][0])},
        }
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet")
    ap.add_argument("--rescore", help="Score a saved --dump file instead of running the sanitizer")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--mode", default="prosan", choices=("prosan", "pii_only"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dump", help="Write every (text, entities, output) to this JSONL file")
    args = ap.parse_args()

    latency = None
    if args.rescore:
        records = [json.loads(l) for l in open(args.rescore, encoding="utf-8") if l.strip()]
    else:
        import pyarrow.parquet as pq
        from sanitizer import PromptSanitizer

        rows = pq.read_table(args.parquet).to_pylist()
        rows = random.Random(args.seed).sample(rows, min(args.n, len(rows)))
        sanitizer = PromptSanitizer(device=args.device)
        sanitizer.sanitize("Warm up the models, John.", evaluate=False)
        records, latencies = [], []
        dump = open(args.dump, "w", encoding="utf-8") if args.dump else None
        for row in rows:
            entities = ast.literal_eval(row["entities"]) if isinstance(row["entities"], str) else row["entities"]
            t0 = time.perf_counter()
            output = sanitizer.sanitize(row["text"], evaluate=False, config_overrides={"mode": args.mode}).text
            latencies.append(time.perf_counter() - t0)
            rec = {"text": row["text"], "entities": entities, "output": output}
            records.append(rec)
            if dump:
                dump.write(json.dumps(rec, ensure_ascii=False) + "\n")
        latency = sum(latencies) / len(latencies)

    result = score(records)
    result["latency_mean_s"] = latency
    result["examples"] = records[:3]
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"{len(records)} documents" + (f", {latency:.3f} s/doc" if latency else ""))
    for mode in ("exact", "strict"):
        r = result[mode]
        print(f"[{mode}] PII leakage rate (Rap-LI Def. 6, mean over docs): {r['pii_leakage_rate_mean']:.2f}%   "
              f"micro: {r['pii_leakage_rate_micro']:.2f}%   entities fully hidden: {r['entities_hidden']:.2f}%")
    for kind, v in list(result["exact"]["by_type"].items())[:20]:
        print(f"  {kind:<28} exact {v['hidden_pct']:6.1f}%   strict {result['strict']['by_type'][kind]['hidden_pct']:6.1f}%  (n={v['n']})")


if __name__ == "__main__":
    main()
