import shutil

import pytest
from PIL import Image, ImageDraw

from sanitizer.image import (
    ImageSanitizer,
    detect_faces,
    detect_text_blocks,
    redact_region,
    strip_metadata,
)

try:
    import cv2  # noqa: F401
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

try:
    import pytesseract  # noqa: F401
    _HAS_OCR = shutil.which("tesseract") is not None
except ImportError:
    _HAS_OCR = False


def _blank(size=(200, 120), color=(255, 255, 255)) -> Image.Image:
    return Image.new("RGB", size, color)


def _text_image(text: str, size=(500, 120)) -> Image.Image:
    image = _blank(size)
    ImageDraw.Draw(image).text((10, 40), text, fill=(0, 0, 0))
    return image


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


@pytest.mark.skipif(not _HAS_OCR, reason="pytesseract/tesseract binary not installed")
def test_detect_text_blocks_reads_rendered_text():
    blocks = detect_text_blocks(_text_image("HELLO WORLD"))
    assert any("HELLO" in text.upper() for _, text in blocks)


def test_detect_text_blocks_returns_empty_without_tesseract(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _no_pytesseract(name, *args, **kwargs):
        if name == "pytesseract":
            raise ImportError("simulated: pytesseract not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_pytesseract)
    assert detect_text_blocks(_blank()) == []


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


@pytest.mark.skipif(not _HAS_OCR, reason="pytesseract/tesseract binary not installed")
def test_sanitize_redacts_email_in_image_text():
    image = _text_image("Contact me at test@example.com please")
    result = ImageSanitizer(redact_faces=False).sanitize(image)

    assert result.text_regions_redacted >= 1
    assert "EMAIL" in result.pii_kinds_found


@pytest.mark.skipif(not _HAS_OCR, reason="pytesseract/tesseract binary not installed")
def test_sanitize_leaves_benign_image_text_untouched():
    image = _text_image("have a nice day")
    result = ImageSanitizer(redact_faces=False).sanitize(image)

    assert result.text_regions_redacted == 0
    assert result.pii_kinds_found == []


def test_sanitize_file_roundtrip(tmp_path):
    src = tmp_path / "in.png"
    _blank().save(src)

    result = ImageSanitizer(redact_faces=False, redact_text=False).sanitize_file(str(src))
    out = tmp_path / "out.png"
    result.save(str(out))

    assert out.exists()
    with Image.open(out) as reopened:
        assert reopened.size == (200, 120)


def test_sanitize_warns_when_tesseract_binary_missing(monkeypatch):
    pytesseract = pytest.importorskip("pytesseract")

    def _missing():
        raise pytesseract.TesseractNotFoundError()

    monkeypatch.setattr(pytesseract, "get_tesseract_version", _missing)
    monkeypatch.setattr(pytesseract, "image_to_data", lambda *a, **kw: (_ for _ in ()).throw(pytesseract.TesseractNotFoundError()))
    result = ImageSanitizer(redact_faces=False).sanitize(_blank())

    assert any("Tesseract OCR binary" in w for w in result.warnings)
