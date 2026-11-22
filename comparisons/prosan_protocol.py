"""Re-create ProSan's privacy evaluation protocol and measure ProSan+ on it.

ProSan (Shen et al., IEEE TIFS 2026, Sec. VI-A) samples 1,000 prompts from each
of MedQA, SAMSum and CodeAlpaca and inserts task-irrelevant sensitive
attributes at random positions:

  Question answering (MedQA)   name, occupation, sexual orientation, birth date
  Summarization (SAMSum)       name, race, location, date   (the transcriber's)
  Code generation (CodeAlpaca) username, password, corporate name, system configuration

ProSan's insertion templates are not published, so this script writes its own
(several per task, randomly chosen and placed). The prompts are therefore
*re-created*, not identical to ProSan's.

PHR follows ProSan's definition, N_hidden / N_total over inserted items, with two
readings of "hidden":
  lenient  the item no longer appears verbatim (ProSan's "removed")
  strict   no content token of the item survives (stop words and generic words
           such as "GB" or "server" excluded), and no half of a credential-like
           token survives contiguously
"Content preserved" is the share of the prompt's own words that survive; it is a
rough utility proxy, not ProSan's LLM-based utility.

    python comparisons/prosan_protocol.py --data-dir DIR --out results.json [--n 1000]

DIR must hold phrases_no_exclude_test.jsonl (GBaker/MedQA-USMLE-4-options),
test.csv (knkarthick/samsum) and code_alpaca_20k.json (sahil2801/CodeAlpaca-20k),
all public on the Hugging Face Hub.
"""

import argparse
import csv
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FIRST = ["James", "Maria", "Wei", "Aisha", "Carlos", "Priya", "Olga", "Kwame", "Sofia", "Hiroshi",
         "Fatima", "Liam", "Ana", "Dmitri", "Chloe", "Rahul", "Ingrid", "Mateo", "Yuki", "Grace",
         "Omar", "Elena", "Tomasz", "Nadia", "Kenji", "Amara", "Lucas", "Mei", "Arjun", "Zara",
         "Henrik", "Leila", "Diego", "Hannah", "Sanjay", "Freya", "Malik", "Isabel", "Pavel", "Ruth"]
LAST = ["Okafor", "Lindqvist", "Nakamura", "Fernandes", "Kowalski", "Haddad", "Mensah", "Petrov",
        "Castillo", "Iyer", "Brennan", "Abernathy", "Moreau", "Varga", "Quintero", "Takahashi",
        "Osei", "Novak", "Rinaldi", "Chaudhry", "Whitfield", "Sorensen", "Delacroix", "Banerjee",
        "Kruger", "Mbeki", "Larsen", "Achterberg", "Yilmaz", "Gallagher"]
OCCUPATIONS = ["farmer", "software engineer", "nurse", "truck driver", "schoolteacher", "accountant",
               "electrician", "pharmacist", "carpenter", "librarian", "police officer", "chef",
               "graphic designer", "plumber", "dental hygienist", "bank teller", "firefighter",
               "real estate agent", "welder", "journalist", "flight attendant", "veterinarian",
               "construction worker", "architect", "social worker", "barista", "lawyer", "mechanic"]
ORIENTATIONS = ["heterosexual", "gay", "lesbian", "bisexual", "pansexual", "asexual", "queer"]
RACES = ["Asian", "Black", "White", "Hispanic", "Latino", "Caucasian", "African American",
         "Native American", "Pacific Islander", "Middle Eastern", "South Asian", "East Asian"]
CITIES = ["Lisbon", "Nairobi", "Osaka", "Toronto", "Krakow", "Lagos", "Hyderabad", "Porto Alegre",
          "Gothenburg", "Adelaide", "Leipzig", "Monterrey", "Izmir", "Cork", "Busan", "Accra",
          "Valparaiso", "Tbilisi", "Brno", "Halifax", "Coimbatore", "Rotterdam", "Tucson", "Perth"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December"]
COMPANIES = ["Northwind Logistics", "Brightwater Analytics", "Kestrel Robotics", "Halcyon Biotech",
             "Silverline Insurance", "Ironbark Mining", "Bluepeak Software", "Corvid Security",
             "Meridian Health Partners", "Tidewater Energy", "Quillon Payments", "Arbor & Finch LLP",
             "Vantage Freight", "Lumen Retail Group", "Obsidian Labs", "Granite Ridge Capital"]
CONFIGS = ["Ubuntu 22.04 on an AWS c5.xlarge instance", "Windows Server 2019 with SQL Server 2017",
           "CentOS 7 on a Dell PowerEdge R740", "Debian 11 running Kubernetes 1.27 on bare metal",
           "macOS Ventura on an M2 MacBook Pro", "RHEL 8.6 with PostgreSQL 14 on Azure D4s_v5",
           "an on-premise Oracle Linux 8 cluster with 128 GB RAM",
           "a Raspberry Pi 4 running Raspbian Bullseye", "Windows 11 Pro with WSL2 Ubuntu 20.04",
           "Alpine Linux 3.18 containers on Google Cloud Run"]

STOP = {"a", "an", "the", "of", "on", "in", "at", "with", "and", "for", "to", "running", "instance",
        "server", "cluster", "gb", "ram", "bare", "metal", "on-premise", "containers", "pro", "linux"}


def birth_date(rng):
    d, m, y = rng.randint(1, 28), rng.randint(1, 12), rng.randint(1950, 2004)
    return rng.choice([f"{MONTHS[m - 1]} {d}, {y}", f"{d} {MONTHS[m - 1]} {y}", f"{m:02d}/{d:02d}/{y}",
                       f"{y}-{m:02d}-{d:02d}"])


def event_date(rng):
    d, m, y = rng.randint(1, 28), rng.randint(1, 12), rng.randint(2019, 2025)
    return rng.choice([f"{MONTHS[m - 1]} {d}, {y}", f"{d} {MONTHS[m - 1]} {y}", f"{m:02d}/{d:02d}/{y}"])


def name(rng):
    return f"{rng.choice(FIRST)} {rng.choice(LAST)}"


def username(rng):
    first, last = rng.choice(FIRST).lower(), rng.choice(LAST).lower()
    return rng.choice([f"{first[0]}{last}{rng.randint(1, 99)}", f"{first}.{last}", f"{first}_{last[:4]}{rng.randint(10, 999)}"])


def password(rng):
    pool = "abcdefghijkmnpqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789!@#$%&*"
    return "".join(rng.choice(pool) for _ in range(rng.randint(9, 14)))


def place(text, insertion, rng):
    """Insert at the start, the end, or a sentence boundary, chosen at random."""
    bounds = [m.end() for m in re.finditer(r"[.?!]\s+", text)]
    where = rng.choice(["start", "end"] + (["middle"] if bounds else []))
    if where == "start":
        return insertion + " " + text
    if where == "end":
        return text.rstrip() + " " + insertion
    cut = rng.choice(bounds)
    return text[:cut] + insertion + " " + text[cut:]


def medqa(row, rng):
    attrs = {"name": name(rng), "occupation": rng.choice(OCCUPATIONS),
             "sexual orientation": rng.choice(ORIENTATIONS), "birth date": birth_date(rng)}
    t = rng.choice([
        "This question was submitted by {name}, a {occupation} born on {birth date} who identifies as {sexual orientation}.",
        "Asked on behalf of {name} ({occupation}; date of birth {birth date}; sexual orientation: {sexual orientation}).",
        "{name} is a {occupation}, was born on {birth date}, and is {sexual orientation}.",
    ]).format(**attrs)
    options = " ".join(f"({k}) {v}" for k, v in sorted(row["options"].items()))
    question = place(row["question"], t, rng)
    return f"{question}\nOptions: {options}\nAnswer with the correct option.", attrs


def samsum(row, rng):
    attrs = {"name": name(rng), "race": rng.choice(RACES), "location": rng.choice(CITIES), "date": event_date(rng)}
    t = rng.choice([
        "This conversation was recorded by a {race} transcriber named {name} in {location} on {date}.",
        "Transcribed by {name} ({race}), {location}, {date}.",
        "Transcriber: {name}, a {race} contractor based in {location}; transcript date {date}.",
    ]).format(**attrs)
    dialogue = row["dialogue"].replace("\r\n", "\n")
    body = f"{t}\n{dialogue}" if rng.random() < 0.5 else f"{dialogue}\n{t}"
    return f"Summarize the following conversation in less than 50 words.\n{body}", attrs


def codealpaca(row, rng):
    attrs = {"username": username(rng), "password": password(rng), "corporate name": rng.choice(COMPANIES),
             "system configuration": rng.choice(CONFIGS)}
    t = rng.choice([
        "This task comes from user {username} at {corporate name}, whose password is {password}; the code will run on {system configuration}.",
        "Context: my login is {username} (password {password}), I work for {corporate name}, and our environment is {system configuration}.",
        "Requested by {username} of {corporate name}. Credentials: password {password}. Target system: {system configuration}.",
    ]).format(**attrs)
    task = place(row["instruction"], t, rng)
    extra = f"\nInput: {row['input']}" if row.get("input") else ""
    return f"Write a program that solves the task below.\nTask: {task}{extra}", attrs


def load(data_dir: Path, n: int, seed: int):
    rng = random.Random(seed)
    med = [json.loads(l) for l in (data_dir / "phrases_no_exclude_test.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    with open(data_dir / "test.csv", encoding="utf-8") as fh:
        sam = [r for r in csv.DictReader(fh) if r.get("dialogue")]
    code = json.loads((data_dir / "code_alpaca_20k.json").read_text(encoding="utf-8"))
    tasks = {"Question answering": (med, medqa), "Summarization": (sam, samsum), "Code generation": (code, codealpaca)}
    out = {}
    for task, (rows, build) in tasks.items():
        picked = rng.sample(rows, min(n, len(rows)))
        out[task] = [build(r, rng) for r in picked]
    return out


_EDGE = ".,;:!?-/'\"()"


def _tokens(text):
    # Strip sentence punctuation from both ends so "bisexual." matches "bisexual"; the same
    # stripping applies to items and outputs, so credential tokens are compared consistently.
    return [t.strip(_EDGE) for t in re.findall(r"[\w#$%&*!@.+/-]+", text.lower()) if t.strip(_EDGE)]


def _token_survives(token: str, out_tokens: set, out_lower: str) -> bool:
    if token in out_tokens:
        return True
    # A credential-like token (digits or symbols) also leaks if half of it survives
    # contiguously: "5K$Hz#nsWHFJ" -> "8N$Hz#nsWHFJ" is not hidden.
    if len(token) >= 6 and not token.isalpha():
        k = (len(token) + 1) // 2
        return any(token[i:i + k] in out_lower for i in range(len(token) - k + 1))
    return False


def hidden(item: str, output: str, strict: bool) -> bool:
    if not strict:
        pattern = (r"(?<!\w)" if item[:1].isalnum() else "") + re.escape(item) + (r"(?!\w)" if item[-1:].isalnum() else "")
        return re.search(pattern, output, re.I) is None
    out_tokens, out_lower = set(_tokens(output)), output.lower()
    content = [t for t in _tokens(item) if t not in STOP]
    return not any(_token_survives(t, out_tokens, out_lower) for t in content)


def content_preserved(prompt: str, attrs: dict, output: str) -> float:
    """Share of the prompt's own words (outside the inserted attributes) that survive."""
    inserted = set()
    for value in attrs.values():
        inserted.update(_tokens(value))
    words = [t for t in _tokens(prompt) if t not in inserted]
    out_tokens = set(_tokens(output))
    return sum(t in out_tokens for t in words) / max(1, len(words))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--mode", default="prosan", choices=("prosan", "pii_only"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dump", help="Write every (task, prompt, attributes, output) to this JSONL file")
    args = ap.parse_args()
    dump = open(args.dump, "w", encoding="utf-8") if args.dump else None

    from sanitizer import PromptSanitizer

    sanitizer = PromptSanitizer(device=args.device)
    sanitizer.sanitize("Warm up the models, John.", evaluate=False)
    data = load(Path(args.data_dir), args.n, args.seed)

    counts = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))  # task -> attr -> [total, lenient, strict]
    latency, examples, preserved = defaultdict(list), defaultdict(list), defaultdict(list)
    for task, items in data.items():
        for prompt, attrs in items:
            t0 = time.perf_counter()
            output = sanitizer.sanitize(prompt, evaluate=False, config_overrides={"mode": args.mode}).text
            latency[task].append(time.perf_counter() - t0)
            preserved[task].append(content_preserved(prompt, attrs, output))
            if dump:
                dump.write(json.dumps({"task": task, "prompt": prompt, "attributes": attrs, "output": output},
                                      ensure_ascii=False) + "\n")
            for attr, value in attrs.items():
                c = counts[task][attr]
                c[0] += 1
                c[1] += hidden(value, output, strict=False)
                c[2] += hidden(value, output, strict=True)
            if len(examples[task]) < 3:
                examples[task].append({"prompt": prompt, "output": output})
        print(f"{task}: {len(items)} prompts, {sum(latency[task]) / len(items):.3f} s/prompt", flush=True)

    result = {"n_per_task": args.n, "seed": args.seed, "mode": args.mode, "tasks": {}, "examples": examples}
    all_total = all_len = all_str = 0
    for task, attrs in counts.items():
        rows = {a: {"n": c[0], "phr_lenient": 100 * c[1] / c[0], "phr_strict": 100 * c[2] / c[0]} for a, c in attrs.items()}
        tot = sum(c[0] for c in attrs.values())
        len_ = sum(c[1] for c in attrs.values())
        str_ = sum(c[2] for c in attrs.values())
        all_total, all_len, all_str = all_total + tot, all_len + len_, all_str + str_
        result["tasks"][task] = {"attributes": rows, "phr_lenient": 100 * len_ / tot, "phr_strict": 100 * str_ / tot,
                                 "content_preserved": 100 * sum(preserved[task]) / len(preserved[task]),
                                 "latency_mean_s": sum(latency[task]) / len(latency[task])}
    result["overall"] = {"phr_lenient": 100 * all_len / all_total, "phr_strict": 100 * all_str / all_total}
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    for task, r in result["tasks"].items():
        print(f"\n{task}  (lenient {r['phr_lenient']:.1f} / strict {r['phr_strict']:.1f}; "
              f"content preserved {r['content_preserved']:.1f}%; {r['latency_mean_s']:.3f} s/prompt)")
        for attr, a in r["attributes"].items():
            print(f"  {attr:<22} lenient {a['phr_lenient']:6.1f}   strict {a['phr_strict']:6.1f}   (n={a['n']})")
    print(f"\nOverall: lenient {result['overall']['phr_lenient']:.1f}  strict {result['overall']['phr_strict']:.1f}")


if __name__ == "__main__":
    main()
