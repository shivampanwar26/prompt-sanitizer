# Text and image test cases

There are 25 labelled text prompts and 25 labelled images. They cover three privacy levels and a
deliberate spread of edge cases. All names, numbers, addresses and keys are fictional. Card,
Aadhaar and IBAN values pass their checksums, so detectors are tested realistically.

| Level | Meaning | Text | Images |
|---|---|---|---|
| `none` | nothing private; the output must be **unchanged** | 8 | 8 |
| `medium` | some private items mixed with content the task needs | 9 | 9 |
| `full` | almost every field identifies someone | 8 | 8 |

## Files

```
text_cases.jsonl     25 prompts
image_cases.jsonl    25 image labels (written by generate_images.py)
images/              the 25 images
generate_images.py   regenerates images/ and image_cases.jsonl
sources/astronaut.png  source photo for every face (public domain, see below)
```

### `text_cases.jsonl`

```json
{"id": "T14", "privacy_level": "medium", "category": "government",
 "edge_case": "Aadhaar number with a valid Verhoeff check digit and an Indian mobile number",
 "prompt": "...", "sensitive": ["4873 1562 0943", "+91 98765 43210"], "keep": ["passport", "police verification"]}
```

- `sensitive` lists the strings that must not appear in the output.
- `keep` lists the strings the task depends on, which must survive.
- The format matches `data/benchmark.jsonl`, so `benchmark.py` runs it directly.

### `image_cases.jsonl`

```json
{"id": "I18", "file": "images/I18_id_card.png", "privacy_level": "full", "category": "document",
 "edge_case": "...", "faces": 1, "face_boxes": [[l, t, r, b]],
 "sensitive_text": ["ELENA MARKOVA", "..."], "keep_text": [], "metadata": []}
```

- `face_boxes` are ground-truth boxes in pixels.
- `sensitive_text` must not be readable by OCR after sanitization.
- `keep_text` must still be readable.
- `metadata` lists the metadata fields that must be reported as removed. The names are the ones
  `strip_metadata` uses.

## Edge cases covered

**Text:**
- **Public entities:** landmarks, fictional characters, nationality words that don't describe
  the user.
- **Look-alikes:** capitalized virtues that look like names; version numbers and error codes that
  look like IPs or IDs.
- **Placeholders:** text that is already anonymized must stay as it is.
- **Unusual names:** all-lowercase names with emoji; a name, then its first name and surname used
  separately; accented Spanish.
- **Checksummed IDs:** Aadhaar, IBAN, a payment card with CVV and expiry, SSN and driver's licence.
- **Secrets:** a password inside code, a `.env` dump, a URL token, a JWT.
- **Hard layouts:** a log line with IP and email, a multi-line contact block, an all-caps
  shipping address, a pipe-separated patient record, a multi-turn chat with a username.

**Images:**
- **No-privacy content:** scenery, a chart, code, a meme, signage, a handwritten-style note.
- **Look-alikes:** a cartoon smiley that must not count as a face; a person whose face is out of
  frame.
- **Faces:** a portrait, two faces, a tiny face of about 19 px (below the detector minimum), a face
  rotated 90°.
- **Metadata:** camera EXIF and GPS, a harmless photo whose only private data is metadata, an
  address hidden in a PNG comment chunk.
- **Partial text:** chat, receipt and terminal screenshots where only one line is private.
- **Identity documents:** an ID card, business card, payment card and shipping label.
- **Hard cases:** a `.env` screenshot, a grayscale blurred JPEG scan, a passport page with
  machine-readable lines, a team photo with name captions.

## Running

```bash
python benchmark.py --data data/testcases/text_cases.jsonl --show   # text, with per-level breakdown
python run_image_cases.py --show                                     # images
python data/testcases/generate_images.py                             # rebuild the images
```

`run_image_cases.py` matches detected faces to `face_boxes` at IoU ≥ 0.3. It checks that the
expected metadata is reported removed and that none is left in the output. It also checks that
`none` cases come back pixel-identical. The text checks re-run OCR on the sanitized image with
the same engine the sanitizer used (Tesseract if installed, otherwise RapidOCR). A reader that is
also the redactor can share its blind spots, so treat the text figures as an upper bound. With no
OCR engine the text checks are reported as **skipped**, never as passed.

## Results (CPU, default config)

**Text** (`benchmark.py`):

| Level | PHR | Utility | Benign intact | Collateral |
|---|---|---|---|---|
| none | n/a | 54.5% | 37.5% | 13.8% |
| medium | 100.0% | 90.0% | n/a | 5.4% |
| full | 90.6% | 100.0% | n/a | 11.7% |
| **all** | **93.9%** | **79.3%** | **37.5%** | **9.6%** |

What these numbers mean:
- **Public names are rewritten.** Only 3 of 8 no-privacy prompts come back unchanged; public
  landmarks and fictional names ("Eiffel Tower", "Harry Potter") and tool names ("pip",
  "nginx") are replaced.
- **"Wisdom, Patience and Discipline" (T02)** is read as a firm-name list and replaced.
- **Some full-level items are missed:** the CVV, the card expiry, and the username
  `tomas.herrera88`.
- **One keep item is lost:** the drug name "amlodipine" (T09) is rewritten.

This set was written after the system was tuned on Presidio. One fix was made because of it: the
all-caps shipping address (T22) used to swallow its `SHIP TO:` label. Apart from that, the set has
not been used for tuning.

**Images** (`run_image_cases.py`, OCR engine RapidOCR, default `replace` style):

| Level | Face recall | Face precision | Metadata removed | Sensitive text hidden | Keep text legible | Unchanged |
|---|---|---|---|---|---|---|
| none | n/a | n/a | n/a | n/a | 13/13 | 8/8 |
| medium | 3/5 | 3/3 | 3/3 | 4/4 | 9/9 | n/a |
| full | 5/5 | 5/5 | 2/2 | 20/30 | 3/3 | n/a |
| **all** | **8/10** | **8/8** | **5/5** | **24/34** | **25/25** | **8/8** |

The `box` style gives the same text figures; the styles differ only in how a span is drawn.

What these numbers mean:
- **Two faces are missed:** the tiny face (I14) and the rotated face (I15). The frontal Haar
  cascade is not built to find them.
- **Text misses are bare identity fields with no pattern or context:**
  - the ID and passport names and numbers (I18, I24), including the passport's machine-readable
    line;
  - the business-card name and address (I19);
  - the card expiry (I20).
- **This set was used during development.** The OCR spacing repair and transcript-level detection
  were developed while inspecting failures on this set, so its text figures are not a blind test.

## Image source and licence

Every face is derived from `sources/astronaut.png`, the scikit-image sample image of NASA
astronaut Eileen Collins. It is a NASA photograph and in the public domain. The faces are
composites of that one photo (scaled, rotated, cropped, grayscaled). The names printed next to
them are fictional and do not refer to the person shown.
