"""Evaluate the image sanitizer on the labelled image test cases.

    python run_image_cases.py                       # data/testcases/image_cases.jsonl
    python run_image_cases.py --show                # one line per case
    python run_image_cases.py --out-dir out/images  # also save the sanitized images

Per case it checks
  faces     detected faces matched to the ground-truth boxes (IoU >= 0.3)
  metadata  every expected metadata field is reported removed, and the output
            carries no EXIF or PNG text chunks
  text      OCR of the *sanitized* image: a sensitive string still readable is
            a leak; a keep string no longer readable is lost
  intact    cases labelled "none" must come back pixel-identical

OCR uses the same engine as the sanitizer (Tesseract if installed, else
RapidOCR). A reader that is the same as the redactor can share its blind
spots, so the text figures are an upper bound on what a different OCR would
find. With no OCR engine the text check is reported as skipped, never as a
pass.
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageChops

from sanitizer.image import ImageSanitizer, detect_faces, detect_text_blocks, ocr_backend, ocr_unavailable_reason

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "testcases" / "image_cases.jsonl"


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def match_faces(truth, found, threshold=0.3):
    """Greedy one-to-one matching; returns (tp, fp, fn)."""
    pairs = sorted(((iou(t, f), i, j) for i, t in enumerate(truth) for j, f in enumerate(found)), reverse=True)
    used_t, used_f = set(), set()
    for score, i, j in pairs:
        if score < threshold or i in used_t or j in used_f:
            continue
        used_t.add(i)
        used_f.add(j)
    tp = len(used_t)
    return tp, len(found) - tp, len(truth) - tp


def norm(text):
    return re.sub(r"[^0-9a-z]+", "", text.lower())


def has_metadata(image):
    return bool(image.getexif()) or any(image.info.get(k) for k in ("exif", "comment", "description", "icc_profile"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=str(DATA))
    parser.add_argument("--show", action="store_true", help="Print one line per case")
    parser.add_argument("--out-dir", help="Save each sanitized image here")
    parser.add_argument("--text-mode", default="replace", choices=("replace", "box", "blur", "pixelate"))
    args = parser.parse_args()

    data = Path(args.data).resolve()
    rows = [json.loads(line) for line in data.read_text(encoding="utf-8").splitlines() if line.strip()]
    ocr_reason = ocr_unavailable_reason()
    ocr = None
    if ocr_reason is None:
        def ocr(image):
            return "\n".join(text for _, text in detect_text_blocks(image))
    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    sanitizer = ImageSanitizer(text_mode=args.text_mode)
    totals = defaultdict(lambda: defaultdict(int))
    for row in rows:
        with Image.open(data.parent / row["file"]) as img:
            img.load()
            original = img.copy()
            original.info = dict(img.info)
        result = sanitizer.sanitize(original)
        if out_dir:
            result.save(str(out_dir / (Path(row["file"]).stem + ".png")))

        tp, fp, fn = match_faces(row["face_boxes"], detect_faces(original))
        meta_missing = [m for m in row["metadata"] if m not in result.metadata_removed]
        meta_ok = not meta_missing and not has_metadata(result.image)

        leaked, lost = [], []
        if ocr:
            seen = norm(ocr(result.image))
            leaked = [s for s in row["sensitive_text"] if norm(s) and norm(s) in seen]
            lost = [k for k in row["keep_text"] if norm(k) and norm(k) not in seen]

        intact = None
        if row["privacy_level"] == "none":
            intact = ImageChops.difference(original.convert("RGB"), result.image.convert("RGB")).getbbox() is None

        for key in (row["privacy_level"], "all"):
            t = totals[key]
            t["cases"] += 1
            t["tp"] += tp; t["fp"] += fp; t["fn"] += fn
            t["meta_cases"] += bool(row["metadata"]); t["meta_ok"] += bool(row["metadata"]) and meta_ok
            t["clean_out"] += not has_metadata(result.image)
            t["sens"] += len(row["sensitive_text"]); t["leaked"] += len(leaked)
            t["keep"] += len(row["keep_text"]); t["lost"] += len(lost)
            if intact is not None:
                t["benign"] += 1; t["intact"] += intact

        if args.show:
            text = "skipped" if not ocr else f"leaked={leaked or '-'} lost={lost or '-'}"
            print(f"{row['id']} {row['privacy_level']:<6} faces {tp}/{len(row['face_boxes'])} (fp {fp})  "
                  f"meta {'ok' if meta_ok else 'MISSING ' + str(meta_missing)}  text {text}"
                  + ("" if intact is None else f"  intact={intact}"))

    def pct(a, b):
        return f"{100.0 * a / b:.1f}%" if b else "n/a"

    print(f"\n{len(rows)} image cases from {data.name}, text mode '{args.text_mode}'")
    if ocr_reason:
        print(f"Text checks skipped: {ocr_reason}.")
    else:
        print(f"OCR engine: {ocr_backend()}")
    print(f"\n{'Level':<8}{'Cases':>6}{'Face recall':>13}{'Face prec.':>12}{'Metadata':>10}"
          f"{'Text PHR':>10}{'Text kept':>11}{'Intact':>8}")
    for level in [*dict.fromkeys(r["privacy_level"] for r in rows), "all"]:
        t = totals[level]
        text_phr = pct(t["sens"] - t["leaked"], t["sens"]) if ocr else "skipped"
        text_kept = pct(t["keep"] - t["lost"], t["keep"]) if ocr else "skipped"
        print(f"{level:<8}{t['cases']:>6}{pct(t['tp'], t['tp'] + t['fn']):>13}{pct(t['tp'], t['tp'] + t['fp']):>12}"
              f"{pct(t['meta_ok'], t['meta_cases']):>10}{text_phr:>10}{text_kept:>11}{pct(t['intact'], t['benign']):>8}")
    print(f"\nOutputs with no metadata left: {totals['all']['clean_out']}/{len(rows)}")


if __name__ == "__main__":
    main()
