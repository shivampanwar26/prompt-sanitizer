import pytest
from PIL import Image, ImageChops, ImageDraw, ImageFont

import sanitizer.image as image_module
from sanitizer.image import (
    ImageSanitizer,
    OcrLine,
    _respace,
    detect_faces,
    detect_text_blocks,
    ocr_backend,
    redact_region,
    replace_text,
    strip_metadata,
)

try:
    import cv2  # noqa: F401
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

_HAS_OCR = ocr_backend() is not None


def _blank(size=(200, 120), color=(255, 255, 255)) -> Image.Image:
    return Image.new("RGB", size, color)


def _text_image(text: str, size=(900, 120)) -> Image.Image:
    image = _blank(size)
    ImageDraw.Draw(image).text((20, 40), text, fill=(0, 0, 0), font=ImageFont.load_default(size=32))
    return image


def _no_ocr(monkeypatch):
    monkeypatch.setattr(image_module, "_tesseract_ready", lambda: False)
    monkeypatch.setattr(image_module, "_rapidocr_ready", lambda: False)


# ── pure, dependency-free mechanics ─────────────────────────────

def test_redact_region_box_changes_only_that_region():
    image = _blank((100, 100), color=(255, 255, 255))
    box = (10, 10, 40, 40)
    redact_region(image, box, mode="box")

    assert image.getpixel((20, 20)) == (0, 0, 0)     # inside the box: blacked out
    assert image.getpixel((80, 80)) == (255, 255, 255)  # outside: untouched


def test_redact_region_pixelate_reduces_detail():
    image = Image.new("RGB", (40, 40))
    pixels = image.load()
    for x in range(40):
        for y in range(40):
            pixels[x, y] = (x * 6 % 256, y * 6 % 256, 0)  # a fine gradient
    before_unique = len(set(image.crop((0, 0, 40, 40)).getdata()))

    redact_region(image, (0, 0, 40, 40), mode="pixelate", strength=10)
    after_unique = len(set(image.crop((0, 0, 40, 40)).getdata()))

    assert after_unique < before_unique


def test_redact_region_rejects_unknown_mode():
    with pytest.raises(ValueError):
        redact_region(_blank(), (0, 0, 10, 10), mode="not-a-mode")


def test_strip_metadata_removes_camera_tags():
    image = _blank()
    exif = Image.Exif()
    exif[272] = "Pixel 9"   # Model
    exif[305] = "GIMP"      # Software
    image.info["exif"] = exif.tobytes()

    clean, removed = strip_metadata(image)

    assert "Model" in removed
    assert "Software" in removed
    assert not clean.getexif()


def test_strip_metadata_removes_gps_location():
    image = _blank()
    exif = Image.Exif()
    gps_ifd = exif.get_ifd(0x8825)   # GPSInfo: a nested IFD, not a flat tag
    gps_ifd[18] = "WGS-84"           # GPSMapDatum (a plain ASCII GPS field)
    image.info["exif"] = exif.tobytes()

    clean, removed = strip_metadata(image)

    assert "GPS location" in removed
    assert not clean.getexif()


def test_strip_metadata_is_a_noop_report_when_nothing_present():
    clean, removed = strip_metadata(_blank())
    assert removed == []
    assert clean.size == (200, 120)


# ── detectors: real when the optional dependency is present ────

def test_detect_faces_returns_list_type_without_crashing():
    # A blank image has no faces; this exercises the real cv2 cascade when
    # opencv is installed, and the graceful "[]" fallback when it is not.
    faces = detect_faces(_blank())
    assert isinstance(faces, list)
    if not _HAS_CV2:
        assert faces == []


@pytest.mark.skipif(not _HAS_OCR, reason="no OCR engine (Tesseract or RapidOCR) installed")
def test_detect_text_blocks_reads_rendered_text():
    blocks = detect_text_blocks(_text_image("HELLO WORLD"))
    assert any("HELLO" in text.upper() for _, text in blocks)


def test_detect_text_blocks_returns_empty_without_any_ocr_engine(monkeypatch):
    _no_ocr(monkeypatch)
    assert detect_text_blocks(_text_image("HELLO WORLD")) == []


def test_respace_restores_spaces_ocr_dropped():
    def line(text):
        return OcrLine((0, 0, 10 * len(text), 20), text, [(10 * i, 0, 10 * i + 10, 20) for i in range(len(text))])

    assert _respace(line("CallRahulat+919820012345aboutthedemo")).text == "Call Rahulat +919820012345 aboutthedemo"
    assert _respace(line("NAME:ELENA MARKOVA")).text == "NAME: ELENA MARKOVA"
    assert _respace(line("DearMr.HarutoSato,")).text == "Dear Mr. Haruto Sato,"
    # Codes, keys, emails, URLs and short tokens are left alone.
    for text in ("1Z999AA10123456784", "sk_live_51HxQ2eK8pL3mN9vR7tY", "jane.doe@example.com",
                 "https://api.example.com/v1", "McDonald", "X4829-117-305", "02:10"):
        assert _respace(line(text)).text == text
    respaced = _respace(line("Sprint14goals"))
    assert respaced.text == "Sprint 14 goals" and len(respaced.char_boxes) == len(respaced.text)


def test_replace_text_stays_inside_its_box_unless_allowed():
    image = _blank((400, 60))
    painted = replace_text(image, (10, 10, 60, 40), "a much longer replacement")
    assert painted[2] <= 60
    assert ImageChops.difference(image.crop((61, 0, 400, 60)), _blank((339, 60))).getbbox() is None
    assert replace_text(_blank((400, 60)), (10, 10, 60, 40), "a much longer replacement", max_right=400)[2] > 60


def test_unknown_text_mode_is_rejected():
    with pytest.raises(ValueError):
        ImageSanitizer(text_mode="smudge")


# ── ImageSanitizer: end-to-end ───────────────────────────────────

def test_sanitize_strips_metadata_even_with_no_faces_or_text():
    image = _blank()
    image.info["exif"] = Image.Exif().tobytes()

    result = ImageSanitizer(redact_faces=False, redact_text=False).sanitize(image)

    assert result.faces_redacted == 0
    assert result.text_regions_redacted == 0
    assert isinstance(result.image, Image.Image)


def test_sanitize_reports_warning_when_face_dependency_missing(monkeypatch):
    import sanitizer.image as image_module

    monkeypatch.setattr(image_module, "detect_faces", lambda image: [])

    import builtins
    real_import = builtins.__import__

    def _no_cv2(name, *args, **kwargs):
        if name == "cv2":
            raise ImportError("simulated: opencv not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_cv2)
    result = ImageSanitizer(redact_text=False).sanitize(_blank())

    assert any("face redaction skipped" in w for w in result.warnings)


@pytest.mark.skipif(not _HAS_OCR, reason="no OCR engine (Tesseract or RapidOCR) installed")
def test_sanitize_redacts_email_in_image_text():
    image = _text_image("Contact me at test@example.com please")
    result = ImageSanitizer(redact_faces=False, text_mode="box").sanitize(image)

    assert result.text_regions_redacted >= 1
    assert "EMAIL" in result.pii_kinds_found


@pytest.mark.skipif(not _HAS_OCR, reason="no OCR engine (Tesseract or RapidOCR) installed")
def test_sanitize_replaces_only_the_sensitive_span_with_a_fake_value():
    image = _text_image("Send feedback to ana.silva@example.org")
    result = ImageSanitizer(redact_faces=False).sanitize(image)

    assert [r["kind"] for r in result.text_replacements] == ["EMAIL"]
    fake = result.text_replacements[0]["replacement"]
    assert "@" in fake and "ana" not in fake and "silva" not in fake
    # "Send feedback to" (left part of the line) is pixel-identical.
    left = (0, 0, 250, 120)
    assert ImageChops.difference(image.crop(left), result.image.crop(left)).getbbox() is None
    reread = " ".join(text for _, text in detect_text_blocks(result.image))
    assert "ana.silva" not in reread


@pytest.mark.skipif(not _HAS_OCR, reason="no OCR engine (Tesseract or RapidOCR) installed")
def test_sanitize_leaves_benign_image_text_untouched():
    image = _text_image("have a nice day")
    result = ImageSanitizer(redact_faces=False).sanitize(image)

    assert result.text_regions_redacted == 0
    assert result.pii_kinds_found == []
    assert ImageChops.difference(image, result.image).getbbox() is None


def test_sanitize_file_roundtrip(tmp_path):
    src = tmp_path / "in.png"
    _blank().save(src)

    result = ImageSanitizer(redact_faces=False, redact_text=False).sanitize_file(str(src))
    out = tmp_path / "out.png"
    result.save(str(out))

    assert out.exists()
    with Image.open(out) as reopened:
        assert reopened.size == (200, 120)


def test_sanitize_warns_when_no_ocr_engine_is_available(monkeypatch):
    _no_ocr(monkeypatch)
    result = ImageSanitizer(redact_faces=False).sanitize(_blank())

    assert any("no OCR engine is available" in w for w in result.warnings)
    assert result.ocr_engine is None


def test_tesseract_is_preferred_and_rapidocr_is_the_fallback(monkeypatch):
    monkeypatch.setattr(image_module, "_tesseract_ready", lambda: True)
    monkeypatch.setattr(image_module, "_rapidocr_ready", lambda: True)
    assert ocr_backend() == "tesseract"
    assert ocr_backend("rapidocr") == "rapidocr"
    monkeypatch.setattr(image_module, "_tesseract_ready", lambda: False)
    assert ocr_backend() == "rapidocr"
    assert ocr_backend("tesseract") is None
