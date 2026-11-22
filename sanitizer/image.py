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
   ``opencv-python-headless`` (``cv2``), which ships the cascade XML inside
   the wheel, so no network download is needed at runtime.
3. **In-image text redaction** — Tesseract OCR (via ``pytesseract``) reads
   text blocks off the image; each block's transcript is run through the
   *same* deterministic detection layer the text pipeline uses
   (:func:`sanitizer.pii.find_pii`), so a screenshot's email address or phone
   number is redacted but its surrounding, non-sensitive text is left
   legible. Requires the Tesseract binary on ``PATH`` in addition to
   ``pytesseract``; if either is missing, this stage is skipped and reported
   as a warning rather than failing the whole image.

Design note: stage 3 only runs each OCR block through the pattern/context/NER
layer (``find_pii``), not the full contextual LM pipeline, to keep per-image
latency low and because OCR transcripts are usually short, isolated phrases
without the surrounding sentence context that layer needs. ``deep=True``
additionally joins every block's transcript and runs it through a full
:class:`~sanitizer.sanitizer.PromptSanitizer` pass for the contextual layer's
extra recall, at the cost of the causal/masked LM latency.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, TYPE_CHECKING

from PIL import Image, ImageDraw, ImageFilter

from .pii import find_pii

if TYPE_CHECKING:
    from .sanitizer import PromptSanitizer

Box = Tuple[int, int, int, int]  # (left, top, right, bottom)

# EXIF tag ids (per the EXIF 2.3 spec) worth naming individually in the report.
_EXIF_GPS_TAG = 34853
_EXIF_NAMED_TAGS = {271: "Make", 272: "Model", 305: "Software", 306: "DateTime", 315: "Artist"}


@dataclass
class ImageSanitizationResult:
    image: Image.Image
    faces_redacted: int = 0
    text_regions_redacted: int = 0
    text_regions_found: List[str] = field(default_factory=list)   # OCR text that was redacted
    pii_kinds_found: List[str] = field(default_factory=list)      # e.g. ["EMAIL", "PHONE"]
    metadata_removed: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def save(self, path: str, **kwargs) -> None:
        self.image.save(path, **kwargs)

    def to_bytes(self, format: str = "PNG") -> bytes:
        buf = io.BytesIO()
        self.image.save(buf, format=format)
        return buf.getvalue()


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


# ── OCR text detection ──────────────────────────────────────────

def ocr_unavailable_reason() -> Optional[str]:
    """Why OCR can't run here, or None if it can (package and binary both present)."""
    try:
        import pytesseract
    except ImportError:
        return "pytesseract is not installed"
    try:
        pytesseract.get_tesseract_version()
    except Exception:
        return "the Tesseract OCR binary is not installed or not on PATH"
    return None


def detect_text_blocks(image: Image.Image, lang: str = "eng", min_confidence: int = 40) -> List[Tuple[Box, str]]:
    """Return [(box, text)] for each OCR'd line, or [] if Tesseract is unavailable."""
    try:
        import pytesseract
    except ImportError:
        return []

    try:
        data = pytesseract.image_to_data(image, lang=lang, output_type=pytesseract.Output.DICT)
    except Exception:
        return []  # pytesseract installed but the Tesseract binary is missing/broken

    lines: dict = {}
    n = len(data.get("text", []))
    for i in range(n):
        word = data["text"][i].strip()
        try:
            conf = int(float(data["conf"][i]))
        except (ValueError, TypeError):
            conf = -1
        if not word or conf < min_confidence:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        l, t = data["left"][i], data["top"][i]
        r, b = l + data["width"][i], t + data["height"][i]
        if key not in lines:
            lines[key] = {"words": [], "box": [l, t, r, b]}
        lines[key]["words"].append(word)
        box = lines[key]["box"]
        box[0], box[1] = min(box[0], l), min(box[1], t)
        box[2], box[3] = max(box[2], r), max(box[3], b)

    return [(tuple(v["box"]), " ".join(v["words"])) for v in lines.values()]


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
        text_mode: str = "box",
        face_margin: float = 0.15,
        ocr_lang: str = "eng",
    ):
        self.redact_faces = redact_faces
        self.redact_text = redact_text
        self.strip_meta = strip_meta
        self.face_mode = face_mode
        self.text_mode = text_mode
        self.face_margin = face_margin
        self.ocr_lang = ocr_lang

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
                result.warnings.append("face redaction skipped: opencv-python-headless is not installed")

        if self.redact_text:
            blocks = detect_text_blocks(working, lang=self.ocr_lang)
            if not blocks:
                reason = ocr_unavailable_reason()
                if reason:
                    result.warnings.append(f"text redaction skipped: {reason}")

            kinds_seen = set()
            for box, text in blocks:
                spans = find_pii(text)
                if not spans:
                    continue
                redact_region(result.image, box, self.text_mode)
                result.text_regions_redacted += 1
                result.text_regions_found.append(text)
                kinds_seen.update(span.kind for span in spans)
            result.pii_kinds_found = sorted(kinds_seen)

            if deep and blocks:
                ps = prompt_sanitizer
                if ps is None:
                    from .sanitizer import PromptSanitizer

                    ps = PromptSanitizer()
                transcript = "\n".join(text for _, text in blocks)
                deep_result = ps.sanitize(transcript, evaluate=False)
                if deep_result.selected_words:
                    extra_kinds = {w["pos_tag"].removeprefix("PII:") for w in deep_result.selected_words}
                    newly_found = extra_kinds - set(result.pii_kinds_found)
                    for box, text in blocks:
                        if any(w["word"] in text for w in deep_result.selected_words):
                            redact_region(result.image, box, self.text_mode)
                    if newly_found:
                        result.warnings.append(
                            f"deep pass found additional sensitive content not caught by OCR-block "
                            f"detection alone: {', '.join(sorted(newly_found))}"
                        )

        if self.strip_meta:
            result.image, result.metadata_removed = strip_metadata(result.image)

        return result

    def sanitize_file(self, path: str, **kwargs) -> ImageSanitizationResult:
        with Image.open(path) as img:
            img.load()
            return self.sanitize(img, **kwargs)
