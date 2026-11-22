"""Image sanitization: the same privacy goal as the text pipeline, applied to
pictures.

Three independent redaction stages, each individually toggleable and each
degrading gracefully (never raising) when its optional dependency is
unavailable — the same philosophy already used for NER/WordNet in
``pii.py``/``nltk_resources.py``:

1. **Metadata stripping** — EXIF (GPS coordinates, device serial, owner
   fields, timestamps), PNG text chunks and ICC profiles are dropped by
   re-encoding the image from raw pixel data alone. Pure Pillow, no optional
   dependency, always runs.
2. **Face redaction** — OpenCV's bundled Haar cascade detects frontal faces;
   each detected region is blurred/pixelated/boxed out. Requires
   ``opencv-python`` 4.x (``cv2``), which ships the cascade XML inside the
   wheel, so no network download is needed at runtime.
3. **In-image text sanitization** — OCR reads each line of text with
   per-character positions. Each line is run through the *same* deterministic
   detection layer the text pipeline uses (:func:`sanitizer.pii.find_pii`), and
   only the characters of each sensitive span are touched, so "Send feedback
   to ana@x.org" keeps "Send feedback to" legible. By default the span is
   painted over in the local background colour and a realistic surrogate from
   the text pipeline's :class:`SurrogateGenerator` is drawn in the original ink
   colour, so the image reads naturally with fake values; ``text_mode`` can
   instead box, blur or pixelate the span.

   Two OCR engines are supported. Tesseract (``pytesseract`` + the Tesseract
   binary) is used when present; otherwise RapidOCR (``rapidocr_onnxruntime``,
   a pip-only package whose models ship inside the wheel) runs offline. If
   neither is available the stage is skipped and reported as a warning.

Design note: stage 3 only runs each OCR line through the pattern/context/NER
layer (``find_pii``), not the full contextual LM pipeline, to keep per-image
latency low and because OCR transcripts are usually short, isolated phrases
without the surrounding sentence context that layer needs. ``deep=True``
additionally joins every line's transcript and runs it through a full
:class:`~sanitizer.sanitizer.PromptSanitizer` pass for the contextual layer's
extra recall, at the cost of the causal/masked LM latency.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .pii import find_pii
from .surrogates import SurrogateGenerator

if TYPE_CHECKING:
    from .sanitizer import PromptSanitizer

Box = Tuple[int, int, int, int]  # (left, top, right, bottom)

# EXIF tag ids (per the EXIF 2.3 spec) worth naming individually in the report.
_EXIF_GPS_TAG = 34853
_EXIF_NAMED_TAGS = {271: "Make", 272: "Model", 305: "Software", 306: "DateTime", 315: "Artist"}

TEXT_MODES = ("replace", "box", "blur", "pixelate")
OCR_ENGINES = ("auto", "tesseract", "rapidocr")


@dataclass
class ImageSanitizationResult:
    image: Image.Image
    faces_redacted: int = 0
    text_regions_redacted: int = 0                                   # sensitive text spans changed
    text_regions_found: List[str] = field(default_factory=list)      # OCR lines that contained them
    text_replacements: List[Dict[str, str]] = field(default_factory=list)  # {kind, original, replacement}
    pii_kinds_found: List[str] = field(default_factory=list)         # e.g. ["EMAIL", "PHONE"]
    metadata_removed: List[str] = field(default_factory=list)
    ocr_engine: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    def save(self, path: str, **kwargs) -> None:
        self.image.save(path, **kwargs)

    def to_bytes(self, format: str = "PNG") -> bytes:
        buf = io.BytesIO()
        self.image.save(buf, format=format)
        return buf.getvalue()


@dataclass
class OcrLine:
    """One OCR'd line. ``char_boxes`` (when known) has one box per character of ``text``."""
    box: Box
    text: str
    char_boxes: Optional[List[Box]] = None


def _expand(box: Box, margin: float, size: Tuple[int, int]) -> Box:
    """Grow a box by ``margin`` (fraction of its own size) and clamp to the image."""
    l, t, r, b = box
    w, h = r - l, b - t
    dx, dy = int(w * margin), int(h * margin)
    return (max(0, l - dx), max(0, t - dy), min(size[0], r + dx), min(size[1], b + dy))


def redact_region(image: Image.Image, box: Box, mode: str = "pixelate", strength: int = 12) -> None:
    """Blur, pixelate or black out ``box`` in place."""
    l, t, r, b = box
    if r <= l or b <= t:
        return
    region = image.crop(box)

    if mode == "box":
        ImageDraw.Draw(image).rectangle(box, fill=(0, 0, 0))
        return
    if mode == "blur":
        region = region.filter(ImageFilter.GaussianBlur(radius=max(3, strength)))
    elif mode == "pixelate":
        factor = max(1, strength)
        small = region.resize((max(1, region.width // factor), max(1, region.height // factor)), Image.NEAREST)
        region = small.resize(region.size, Image.NEAREST)
    else:
        raise ValueError("mode must be 'blur', 'pixelate' or 'box'")

    image.paste(region, box)


def _luminance(rgb) -> float:
    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]


def _background_and_ink(image: Image.Image, box: Box) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """Estimate the background colour around ``box`` and the text (ink) colour inside it."""
    l, t, r, b = box
    w, h = image.size
    ring = []
    for x in range(max(0, l - 2), min(w, r + 2)):
        for y in (max(0, t - 2), min(h - 1, b + 1)):
            ring.append(image.getpixel((x, y)))
    for y in range(max(0, t - 2), min(h, b + 2)):
        for x in (max(0, l - 2), min(w - 1, r + 1)):
            ring.append(image.getpixel((x, y)))
    if not ring:
        ring = [image.getpixel((min(l, w - 1), min(t, h - 1)))]
    background = tuple(sorted(c[i] for c in ring)[len(ring) // 2] for i in range(3))

    bg_lum = _luminance(background)
    inside = image.crop(box).resize((max(1, (r - l) // 2), max(1, (b - t) // 2)))
    ink = max(inside.getdata(), key=lambda c: abs(_luminance(c) - bg_lum), default=(0, 0, 0))
    if abs(_luminance(ink) - bg_lum) < 60:  # no clear text pixels: pick a contrasting default
        ink = (20, 20, 20) if bg_lum > 128 else (235, 235, 235)
    return background, tuple(ink)


def replace_text(image: Image.Image, box: Box, replacement: str, max_right: Optional[int] = None) -> Box:
    """Paint over ``box`` with its background and draw ``replacement`` in the original ink colour.

    The font shrinks to fit the box. If the replacement is still longer, the
    painted area may grow right, but never past ``max_right`` (default: the
    box's own right edge, so neighbouring text is never covered). Returns the
    box actually painted.
    """
    l, t, r, b = box
    height = max(1, b - t)
    limit = min(image.width, r if max_right is None else max(r, max_right))
    background, ink = _background_and_ink(image, box)
    draw = ImageDraw.Draw(image)

    size = max(8, int(height * 0.8))
    floor = 6 if limit <= r else max(8, int(height * 0.55))
    font = ImageFont.load_default(size=size)
    width = draw.textlength(replacement, font=font)
    while width > (r - l) and size > floor:
        size -= 1
        font = ImageFont.load_default(size=size)
        width = draw.textlength(replacement, font=font)
    right = min(limit, max(r, l + int(width) + 2))

    # Draw on a tile the size of the painted area, so glyphs that still don't
    # fit are clipped instead of spilling onto neighbouring text.
    painted = (l, t, right, b)
    tile = Image.new("RGB", (max(1, right - l), height), background)
    ascent, descent = font.getmetrics()
    ImageDraw.Draw(tile).text((1, (height - (ascent + descent)) // 2), replacement, fill=ink, font=font)
    image.paste(tile, (l, t))
    return painted


def strip_metadata(image: Image.Image) -> Tuple[Image.Image, List[str]]:
    """Return (clean_copy, [names of metadata categories that were removed])."""
    removed = []
    try:
        exif = image.getexif()
    except Exception:
        exif = None

    if exif:
        if _EXIF_GPS_TAG in exif:
            removed.append("GPS location")
        for tag, name in _EXIF_NAMED_TAGS.items():
            if tag in exif:
                removed.append(name)
        if not removed:
            removed.append("EXIF data")
    if image.info.get("icc_profile"):
        removed.append("ICC color profile")
    for key in ("comment", "description"):
        if image.info.get(key):
            removed.append(f"PNG {key} chunk")

    # Re-encoding from raw pixel bytes (rather than image.copy()) drops every
    # ancillary chunk Pillow preserves on copy: exif, icc_profile, text chunks.
    clean = Image.frombytes(image.mode, image.size, image.tobytes())
    return clean, removed


# ── face detection ──────────────────────────────────────────────

_FACE_CASCADE = None


def _face_cascade():
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        import cv2

        _FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return _FACE_CASCADE


def detect_faces(image: Image.Image) -> List[Box]:
    """Return face bounding boxes, or [] if OpenCV is unavailable or none are found."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return []

    cascade = _face_cascade()
    gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
    return [(int(x), int(y), int(x + w), int(y + h)) for x, y, w, h in faces]


# ── OCR ─────────────────────────────────────────────────────────

_RAPID_ENGINE = None


def _tesseract_ready() -> bool:
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def _rapidocr_engine():
    global _RAPID_ENGINE
    if _RAPID_ENGINE is None:
        from rapidocr_onnxruntime import RapidOCR

        _RAPID_ENGINE = RapidOCR()
    return _RAPID_ENGINE


def _rapidocr_ready() -> bool:
    try:
        import rapidocr_onnxruntime  # noqa: F401
        return True
    except Exception:
        return False


def ocr_backend(engine: str = "auto") -> Optional[str]:
    """The OCR engine that will run: "tesseract", "rapidocr", or None if neither can."""
    if engine not in OCR_ENGINES:
        raise ValueError(f"engine must be one of {OCR_ENGINES}")
    if engine in ("auto", "tesseract") and _tesseract_ready():
        return "tesseract"
    if engine in ("auto", "rapidocr") and _rapidocr_ready():
        return "rapidocr"
    return None


def ocr_unavailable_reason(engine: str = "auto") -> Optional[str]:
    """Why OCR can't run here, or None if it can."""
    if ocr_backend(engine):
        return None
    if engine == "tesseract":
        return "the Tesseract OCR binary (or pytesseract) is not installed or not on PATH"
    if engine == "rapidocr":
        return "rapidocr_onnxruntime is not installed"
    return ("no OCR engine is available: install rapidocr_onnxruntime (pip) "
            "or the Tesseract binary plus pytesseract")


def _tesseract_lines(image: Image.Image, lang: str, min_confidence: int) -> List[OcrLine]:
    import pytesseract

    try:
        data = pytesseract.image_to_data(image, lang=lang, output_type=pytesseract.Output.DICT)
    except Exception:
        return []

    lines: dict = {}
    for i in range(len(data.get("text", []))):
        word = data["text"][i].strip()
        try:
            conf = int(float(data["conf"][i]))
        except (ValueError, TypeError):
            conf = -1
        if not word or conf < min_confidence:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        l, t = data["left"][i], data["top"][i]
        lines.setdefault(key, []).append((word, (l, t, l + data["width"][i], t + data["height"][i])))

    out = []
    for words in lines.values():
        top = min(b[1] for _, b in words)
        bottom = max(b[3] for _, b in words)
        text, chars = "", []
        for n, (word, (l, t, r, b)) in enumerate(words):
            if n:
                prev_right = chars[-1][2]
                text += " "
                chars.append((prev_right, top, max(prev_right, l), bottom))
            step = (r - l) / max(1, len(word))
            for k in range(len(word)):
                chars.append((int(l + k * step), top, int(l + (k + 1) * step), bottom))
            text += word
        box = (min(c[0] for c in chars), top, max(c[2] for c in chars), bottom)
        out.append(OcrLine(box, text, chars))
    return out


def _quad_box(quad) -> Box:
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    return (int(min(xs)), int(min(ys)), int(round(max(xs))), int(round(max(ys))))


def _rapidocr_lines(image: Image.Image, min_score: float = 0.5) -> List[OcrLine]:
    import numpy as np

    try:
        result, _ = _rapidocr_engine()(np.array(image.convert("RGB")), return_word_box=True)
    except Exception:
        return []

    out = []
    for entry in result or []:
        quad, raw_text, score = entry[0], entry[1], float(entry[2])
        if score < min_score or not raw_text.strip():
            continue
        box = _quad_box(quad)
        tokens = entry[4] if len(entry) > 4 else None
        token_quads = entry[3] if len(entry) > 3 else None
        if not tokens or not token_quads or len(tokens) != len(token_quads):
            out.append(OcrLine(box, raw_text, None))
            continue

        text, chars = "", []
        for token, quad_k in zip(tokens, token_quads):
            tb = _quad_box(quad_k)
            if token.isspace():
                if text and not text.endswith(" "):
                    text += " "
                    chars.append(tb)
                continue
            step = (tb[2] - tb[0]) / max(1, len(token))
            for k, ch in enumerate(token):
                text += ch
                chars.append((int(tb[0] + k * step), box[1], int(tb[0] + (k + 1) * step), box[3]))
        out.append(_respace(OcrLine(box, text, chars)))
    return out


# RapidOCR's recogniser often drops spaces ("CallRahulat+919820012345", "NAME:ELENA").
# Its character boxes are fixed-width decoder frames, so pixel gaps can't tell a space
# from letter spacing; spaces are restored from the text's structure instead, and only
# inside long run-together tokens that don't look like codes, keys, emails or URLs.
_RUN_TOGETHER = 12
_NOT_PROSE = re.compile(r"[@/\\_=<>]|https?|www\.", re.I)
_CAMEL_JOIN = re.compile(r"(?<=[a-z]{3})(?=[A-Z][a-z])|(?<=[a-z]\.)(?=[A-Z])")
_LETTER_DIGIT = re.compile(r"(?<=[A-Za-z])(?=[+\d])|(?<=\d)(?=[A-Za-z])")
_LABEL_JOIN = re.compile(r"(?<=[A-Za-z][:;,])(?=[A-Za-z0-9(+])")


def _respace(line: OcrLine) -> OcrLine:
    text, chars = line.text, line.char_boxes
    cuts = set(m.start() for m in _LABEL_JOIN.finditer(text))
    for token in re.finditer(r"\S+", text):
        word = token.group(0)
        if len(word) < _RUN_TOGETHER or _NOT_PROSE.search(word):
            continue
        cuts.update(token.start() + m.start() for m in _CAMEL_JOIN.finditer(word))
        if re.search(r"[a-z]{3}", word):  # prose, not an all-caps code like 1Z999AA10123456784
            cuts.update(token.start() + m.start() for m in _LETTER_DIGIT.finditer(word))
    if not cuts:
        return line
    new_text, new_chars = "", []
    for i, ch in enumerate(text):
        if i in cuts and i > 0 and not text[i - 1].isspace():
            new_text += " "
            if chars:
                edge = chars[i][0]
                new_chars.append((edge, chars[i][1], edge, chars[i][3]))
        new_text += ch
        if chars:
            new_chars.append(chars[i])
    return OcrLine(line.box, new_text, new_chars if chars else None)


def detect_text_lines(image: Image.Image, lang: str = "eng", engine: str = "auto",
                      min_confidence: int = 40) -> List[OcrLine]:
    """OCR ``image`` into lines with per-character boxes; [] if no engine can run."""
    backend = ocr_backend(engine)
    if backend == "tesseract":
        return _tesseract_lines(image, lang, min_confidence)
    if backend == "rapidocr":
        return _rapidocr_lines(image)
    return []


def detect_text_blocks(image: Image.Image, lang: str = "eng", min_confidence: int = 40,
                       engine: str = "auto") -> List[Tuple[Box, str]]:
    """Return [(box, text)] for each OCR'd line, or [] if no OCR engine is available."""
    return [(line.box, line.text) for line in detect_text_lines(image, lang, engine, min_confidence)]


def _span_box(line: OcrLine, start: int, end: int, size: Tuple[int, int]) -> Box:
    """Pixel box of characters [start, end) of ``line`` (whole line if positions are unknown)."""
    if not line.char_boxes or len(line.char_boxes) != len(line.text):
        return line.box
    boxes = [line.char_boxes[i] for i in range(start, end) if not line.text[i].isspace()]
    if not boxes:
        return line.box
    l, r = min(b[0] for b in boxes), max(b[2] for b in boxes)
    t, b = line.box[1], line.box[3]
    pad_x, pad_y = max(2, (b - t) // 8), max(1, (b - t) // 10)
    return (max(0, l - pad_x), max(0, t - pad_y), min(size[0], r + pad_x), min(size[1], b + pad_y))


# ── orchestration ────────────────────────────────────────────────

class ImageSanitizer:
    """Redacts faces and sensitive in-image text, and strips metadata.

    Mirrors :class:`~sanitizer.sanitizer.PromptSanitizer`'s layered, fail-soft
    design: each stage is independent, and a missing optional dependency
    degrades that stage to a no-op (reported in ``warnings``) rather than
    raising.
    """

    def __init__(
        self,
        redact_faces: bool = True,
        redact_text: bool = True,
        strip_meta: bool = True,
        face_mode: str = "pixelate",
        text_mode: str = "replace",
        face_margin: float = 0.15,
        ocr_lang: str = "eng",
        ocr_engine: str = "auto",
        surrogate_secret: str = "prosan-image",
    ):
        if text_mode not in TEXT_MODES:
            raise ValueError(f"text_mode must be one of {TEXT_MODES}")
        if ocr_engine not in OCR_ENGINES:
            raise ValueError(f"ocr_engine must be one of {OCR_ENGINES}")
        self.redact_faces = redact_faces
        self.redact_text = redact_text
        self.strip_meta = strip_meta
        self.face_mode = face_mode
        self.text_mode = text_mode
        self.face_margin = face_margin
        self.ocr_lang = ocr_lang
        self.ocr_engine = ocr_engine
        self.surrogate_secret = surrogate_secret

    def _apply(self, image: Image.Image, box: Box, replacement: str, max_right: Optional[int]) -> None:
        if self.text_mode == "replace":
            replace_text(image, box, replacement, max_right)
        else:
            redact_region(image, box, self.text_mode)

    def sanitize(
        self,
        image: Image.Image,
        deep: bool = False,
        prompt_sanitizer: Optional["PromptSanitizer"] = None,
    ) -> ImageSanitizationResult:
        """Return a redacted copy of ``image``; the input is never modified."""
        working = image.convert("RGB")
        result = ImageSanitizationResult(image=working.copy())

        if self.redact_faces:
            faces = detect_faces(working)
            if faces:
                for box in faces:
                    redact_region(result.image, _expand(box, self.face_margin, working.size), self.face_mode)
                result.faces_redacted = len(faces)
            try:
                import cv2  # noqa: F401
            except ImportError:
                result.warnings.append("face redaction skipped: opencv-python is not installed")

        if self.redact_text:
            self._sanitize_text(working, result, deep, prompt_sanitizer)

        if self.strip_meta:
            result.image, result.metadata_removed = strip_metadata(result.image)

        return result

    def _sanitize_text(self, working, result, deep, prompt_sanitizer) -> None:
        reason = ocr_unavailable_reason(self.ocr_engine)
        if reason:
            result.warnings.append(f"text redaction skipped: {reason}")
            return
        result.ocr_engine = ocr_backend(self.ocr_engine)
        lines = detect_text_lines(working, lang=self.ocr_lang, engine=self.ocr_engine)
        surrogates = SurrogateGenerator(style="realistic", secret=self.surrogate_secret)

        # Detect over the whole transcript, not line by line, so a bare name under
        # "SHIP TO:" or a postal line under a street gets the same cross-line context
        # the text pipeline has. Spans are then cut back to the lines they cover.
        transcript = "\n".join(line.text for line in lines)
        offsets, pos = [], 0
        for line in lines:
            offsets.append(pos)
            pos += len(line.text) + 1

        # (line index, start, end, kind, replacement); spans never overlap within a line.
        edits: List[Tuple[int, int, int, str, str]] = []
        for span in find_pii(transcript):
            for n, line in enumerate(lines):
                start = max(span.start, offsets[n]) - offsets[n]
                end = min(span.end, offsets[n] + len(line.text)) - offsets[n]
                original = line.text[start:end] if end > start else ""
                if not original.strip():
                    continue
                lead = len(original) - len(original.lstrip())
                start, end = start + lead, start + len(original.rstrip())
                original = line.text[start:end]
                edits.append((n, start, end, span.kind, surrogates.get(span.kind, original)))

        if deep and lines:
            ps = prompt_sanitizer
            if ps is None:
                from .sanitizer import PromptSanitizer

                ps = PromptSanitizer()
            transcript = "\n".join(line.text for line in lines)
            deep_result = ps.sanitize(transcript, evaluate=False)
            found_kinds = {kind for *_, kind, _ in edits}
            new_kinds = set()
            for word in deep_result.selected_words:
                kind = word["pos_tag"].removeprefix("PII:")
                for n, line in enumerate(lines):
                    for m in re.finditer(re.escape(word["word"]), line.text):
                        taken = any(e[0] == n and m.start() < e[2] and m.end() > e[1] for e in edits)
                        if not taken:
                            edits.append((n, m.start(), m.end(), kind, word["replacement"]))
                            if kind not in found_kinds:
                                new_kinds.add(kind)
            if new_kinds:
                result.warnings.append(
                    "deep pass found additional sensitive content not caught by line-level "
                    f"detection alone: {', '.join(sorted(new_kinds))}"
                )

        kinds = set()
        touched_lines = set()
        for n, start, end, kind, replacement in sorted(edits, key=lambda e: (e[0], e[1])):
            line = lines[n]
            # A replacement may only spill right when nothing but whitespace follows it.
            at_line_end = not line.text[end:].strip()
            max_right = working.size[0] if at_line_end else None
            self._apply(result.image, _span_box(line, start, end, working.size), replacement, max_right)
            result.text_regions_redacted += 1
            result.text_replacements.append(
                {"kind": kind, "original": line.text[start:end],
                 "replacement": replacement if self.text_mode == "replace" else f"[{self.text_mode}]"})
            kinds.add(kind)
            if n not in touched_lines:
                touched_lines.add(n)
                result.text_regions_found.append(line.text)
        result.pii_kinds_found = sorted(kinds)

    def sanitize_file(self, path: str, **kwargs) -> ImageSanitizationResult:
        with Image.open(path) as img:
            img.load()
            return self.sanitize(img, **kwargs)
