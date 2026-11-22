"""Regenerate the 25 image test cases and their labels (image_cases.jsonl).

    python data/testcases/generate_images.py

Everything is drawn with Pillow's bundled font, so the output is the same on
every OS. Faces come from one public-domain photograph (see README.md):
the generator pastes, scales, rotates and crops it, and records where the face
ends up, so each case carries ground-truth face boxes for the evaluator.
All names, numbers, addresses and keys are fictional; card and ID numbers are
built to pass their checksums so detectors are tested realistically.
"""

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont
from PIL.PngImagePlugin import PngInfo

HERE = Path(__file__).resolve().parent
OUT = HERE / "images"
SOURCE = HERE / "sources" / "astronaut.png"
FACE_IN_SOURCE = (177, 66, 272, 161)  # face box in the 512x512 source photo

WHITE, BLACK, GREY = (255, 255, 255), (20, 20, 20), (110, 110, 110)


def font(size):
    return ImageFont.load_default(size=size)


def text_lines(draw, lines, xy, size=26, fill=BLACK, gap=1.45):
    x, y = xy
    f = font(size)
    for line in lines:
        draw.text((x, y), line, fill=fill, font=f)
        y += int(size * gap)
    return y


def place_face(canvas, size, pos, crop=(0, 0, 512, 512), rotate=0, gray=False, blur=0):
    """Paste (a crop of) the source photo; return the face box on the canvas, or None."""
    src = Image.open(SOURCE).convert("RGB").crop(crop)
    w, h = size
    sx, sy = w / (crop[2] - crop[0]), h / (crop[3] - crop[1])
    piece = src.resize((w, h), Image.LANCZOS)
    fl, ft, fr, fb = FACE_IN_SOURCE
    l, t = (fl - crop[0]) * sx, (ft - crop[1]) * sy
    r, b = (fr - crop[0]) * sx, (fb - crop[1]) * sy
    face = None if r <= 0 or b <= 0 or l >= w or t >= h else (max(l, 0), max(t, 0), min(r, w), min(b, h))
    if rotate == 90:  # counter-clockwise: (u, v) -> (v, w - u)
        piece = piece.rotate(90, expand=True)
        if face:
            l, t, r, b = face
            face = (t, w - r, b, w - l)
    if gray:
        piece = piece.convert("L").convert("RGB")
    if blur:
        piece = piece.filter(ImageFilter.GaussianBlur(blur))
    canvas.paste(piece, pos)
    if face is None:
        return None
    return [round(face[0] + pos[0]), round(face[1] + pos[1]), round(face[2] + pos[0]), round(face[3] + pos[1])]


def exif(**fields):
    ex = Image.Exif()
    tags = {"make": 271, "model": 272, "software": 305, "datetime": 306, "artist": 315}
    for key, value in fields.items():
        if key == "gps":
            ex[0x8825] = {1: "N", 2: (value[0], value[1], 0.0), 3: "E", 4: (value[2], value[3], 0.0)}
        else:
            ex[tags[key]] = value
    return ex


CASES = []


def case(cid, slug, level, category, edge_case, faces=(), sensitive=(), keep=(), metadata=()):
    CASES.append({
        "id": cid, "file": f"images/{slug}", "privacy_level": level, "category": category,
        "edge_case": edge_case, "faces": len([f for f in faces if f]), "face_boxes": [f for f in faces if f],
        "sensitive_text": list(sensitive), "keep_text": list(keep), "metadata": list(metadata),
    })


def card(size=(900, 520), color=WHITE):
    im = Image.new("RGB", size, color)
    return im, ImageDraw.Draw(im)


def build():
    OUT.mkdir(exist_ok=True)

    # ── no privacy ───────────────────────────────────────────────
    im, d = card((800, 500), (135, 190, 235))
    d.rectangle((0, 330, 800, 500), fill=(90, 150, 80))
    d.polygon([(80, 330), (260, 120), (440, 330)], fill=(120, 120, 130))
    d.polygon([(330, 330), (520, 90), (720, 330)], fill=(100, 100, 115))
    d.ellipse((620, 40, 700, 120), fill=(250, 220, 90))
    im.save(OUT / "I01_landscape.png")
    case("I01", "I01_landscape.png", "none", "photo", "scenery with no people and no text")

    im, d = card((900, 520))
    d.text((40, 30), "Quarterly Revenue Growth", fill=BLACK, font=font(40))
    for i, (q, v) in enumerate([("Q1", 120), ("Q2", 180), ("Q3", 240), ("Q4", 300)]):
        x = 120 + i * 180
        d.rectangle((x, 460 - v, x + 90, 460), fill=(70, 130, 200))
        d.text((x + 25, 470), q, fill=BLACK, font=font(28))
    im.save(OUT / "I02_chart.png")
    case("I02", "I02_chart.png", "none", "slide", "business chart with only public labels",
         keep=["Quarterly Revenue Growth", "Q1", "Q4"])

    im, d = card((900, 320), (30, 33, 40))
    text_lines(d, ["def add(a, b):", "    return a + b", "", "print(add(2, 3))"], (40, 40), 30, (220, 220, 200))
    im.save(OUT / "I03_code.png")
    case("I03", "I03_code.png", "none", "screenshot", "code screenshot with no secrets",
         keep=["def add(a, b):", "return a + b"])

    im, d = card((900, 500), (240, 240, 240))
    d.text((60, 60), "WHEN THE CODE COMPILES", fill=BLACK, font=font(52))
    d.text((60, 380), "ON THE FIRST TRY", fill=BLACK, font=font(52))
    d.ellipse((370, 170, 530, 330), outline=BLACK, width=6)
    im.save(OUT / "I04_meme.png")
    case("I04", "I04_meme.png", "none", "meme", "all-caps meme text that looks like a name line",
         keep=["WHEN THE CODE COMPILES", "ON THE FIRST TRY"])

    im, d = card((900, 420), (20, 90, 60))
    d.text((60, 60), "EXIT", fill=WHITE, font=font(90))
    text_lines(d, ["Platform 3", "Trains to City Centre"], (60, 220), 44, WHITE)
    im.save(OUT / "I05_sign.png")
    case("I05", "I05_sign.png", "none", "photo", "public signage with place words",
         keep=["EXIT", "Platform 3", "City Centre"])

    im, d = card((600, 600), (250, 250, 250))
    d.ellipse((100, 100, 500, 500), fill=(255, 205, 60), outline=BLACK, width=5)
    d.ellipse((210, 220, 260, 290), fill=BLACK)
    d.ellipse((340, 220, 390, 290), fill=BLACK)
    d.arc((190, 260, 410, 420), 20, 160, fill=BLACK, width=10)
    im.save(OUT / "I06_smiley.png")
    case("I06", "I06_smiley.png", "none", "illustration", "cartoon smiley must not count as a human face")

    im, d = card((900, 400), (252, 248, 225))
    text_lines(d, ["Shopping list:", "buy milk, eggs, bread", "call the plumber about the sink"], (50, 60), 34, (95, 95, 120))
    im.save(OUT / "I07_note.png")
    case("I07", "I07_note.png", "none", "note", "low-contrast handwritten-style note with no personal data",
         keep=["Shopping list", "milk, eggs, bread", "plumber"])

    im = Image.new("RGB", (512, 300), WHITE)
    place_face(im, (512, 300), (0, 0), crop=(0, 250, 512, 512))
    im.save(OUT / "I08_person_no_face.png")
    case("I08", "I08_person_no_face.png", "none", "photo", "person in frame but face not visible (cropped)")

    # ── medium privacy ───────────────────────────────────────────
    im = Image.new("RGB", (512, 512))
    f = place_face(im, (512, 512), (0, 0))
    im.save(OUT / "I09_portrait_exif.jpg", quality=95,
            exif=exif(make="Canon", model="EOS 80D", gps=(28, 36, 77, 12)))
    case("I09", "I09_portrait_exif.jpg", "medium", "photo", "portrait photo with camera and GPS EXIF",
         faces=[f], metadata=["GPS location", "Make", "Model"])

    im, d = card((900, 420), (235, 238, 245))
    y = 40
    for who, msg in [("Ravi", "Hi team, the report is ready."), ("Mia", "Great, sharing it today."),
                     ("Ravi", "Send feedback to ana.silva@example.org"), ("Mia", "Thanks!")]:
        d.rounded_rectangle((40, y, 860, y + 70), 14, fill=WHITE)
        d.text((60, y + 18), msg, fill=BLACK, font=font(28))
        y += 90
    im.save(OUT / "I10_chat.png")
    case("I10", "I10_chat.png", "medium", "screenshot", "one email address among benign chat lines",
         sensitive=["ana.silva@example.org"], keep=["the report is ready", "Thanks"])

    im, d = card((600, 760))
    text_lines(d, ["GREEN LEAF CAFE", "", "Cappuccino        4.50", "Bagel             3.20",
                   "Orange juice      5.80", "", "Total: $13.50", "Paid by card **** 4821",
                   "Customer phone: 555-201-7788", "", "Thank you!"], (40, 40), 28)
    im.save(OUT / "I11_receipt.png")
    case("I11", "I11_receipt.png", "medium", "document",
         "receipt: already-masked card digits are fine, the phone number is not",
         sensitive=["555-201-7788"], keep=["Total: $13.50", "Cappuccino", "**** 4821"])

    im, d = card((1000, 330), (12, 12, 12))
    text_lines(d, ["$ cd project && npm test", "All 42 tests passed",
                   "$ export STRIPE_KEY=sk_live_51HxQ2eK8pL3mN9vR7tY",
                   "$ curl https://api.example.com/health"], (30, 40), 28, (90, 230, 120))
    im.save(OUT / "I12_terminal.png")
    case("I12", "I12_terminal.png", "medium", "screenshot", "live API key in a terminal among harmless commands",
         sensitive=["sk_live_51HxQ2eK8pL3mN9vR7tY"], keep=["npm test", "All 42 tests passed"])

    im = Image.new("RGB", (900, 450), (200, 200, 200))
    f1 = place_face(im, (420, 420), (15, 15))
    f2 = place_face(im, (420, 420), (465, 15))
    im.save(OUT / "I13_two_faces.png")
    case("I13", "I13_two_faces.png", "medium", "photo", "two faces in one picture", faces=[f1, f2])

    im = Image.new("RGB", (1200, 800), (180, 210, 230))
    f = place_face(im, (100, 100), (600, 400))
    im.save(OUT / "I14_tiny_face.png")
    case("I14", "I14_tiny_face.png", "medium", "photo",
         "very small face (about 19 px), below the detector's 24 px minimum", faces=[f])

    im = Image.new("RGB", (512, 512))
    f = place_face(im, (512, 512), (0, 0), rotate=90)
    im.save(OUT / "I15_rotated_face.png")
    case("I15", "I15_rotated_face.png", "medium", "photo",
         "face rotated 90 degrees (frontal detectors usually miss it)", faces=[f])

    im, d = card((800, 500), (140, 200, 240))
    d.rectangle((0, 320, 800, 500), fill=(230, 210, 150))
    d.ellipse((560, 50, 660, 150), fill=(255, 230, 100))
    im.save(OUT / "I16_beach_gps.jpg", quality=95,
            exif=exif(gps=(15, 30, 73, 50), datetime="2026:08:14 17:22:05", artist="Neha Kulkarni"))
    case("I16", "I16_beach_gps.jpg", "medium", "photo",
         "harmless picture whose only private data is in metadata (GPS, owner name, time)",
         metadata=["GPS location", "DateTime", "Artist"])

    im, d = card((1000, 560), (248, 248, 248))
    d.rectangle((10, 10, 990, 550), outline=GREY, width=6)
    text_lines(d, ["Sprint 14 goals", "- finish the login page", "- fix flaky upload test",
                   "Call Rahul at +91 98200 12345 about the demo"], (50, 50), 36, (30, 60, 160))
    meta = PngInfo()
    meta.add_text("comment", "Whiteboard, Floor 3, 221B Baker Street office")
    im.save(OUT / "I17_whiteboard.png", pnginfo=meta)
    case("I17", "I17_whiteboard.png", "medium", "photo",
         "whiteboard notes with one phone number, plus an address in a PNG comment chunk",
         sensitive=["+91 98200 12345"], keep=["Sprint 14 goals", "finish the login page"],
         metadata=["PNG comment chunk"])

    # ── full privacy ─────────────────────────────────────────────
    im, d = card((1000, 620), (225, 235, 245))
    d.rectangle((0, 0, 1000, 90), fill=(30, 70, 140))
    d.text((30, 25), "NATIONAL IDENTITY CARD", fill=WHITE, font=font(40))
    f = place_face(im, (260, 300), (40, 130), crop=(110, 10, 340, 275))
    text_lines(d, ["NAME: ELENA MARKOVA", "DOB: 12/03/1988", "ID NO: X4829-117-305",
                   "ADDRESS: 44 Birch Lane", "Leeds LS6 2QT"], (340, 140), 34)
    im.save(OUT / "I18_id_card.png")
    case("I18", "I18_id_card.png", "full", "document", "ID card: photo plus name, date of birth, ID number, address",
         faces=[f], sensitive=["ELENA MARKOVA", "12/03/1988", "X4829-117-305", "44 Birch Lane", "LS6 2QT"])

    im, d = card((900, 500))
    d.rectangle((0, 0, 18, 500), fill=(200, 60, 60))
    text_lines(d, ["Arjun Kapoor", "Senior Data Engineer", "", "+44 20 7946 0958",
                   "arjun.kapoor@northwind.co.uk", "12 King's Road, London SW3 4RP"], (60, 50), 34)
    im.save(OUT / "I19_business_card.png")
    case("I19", "I19_business_card.png", "full", "document", "business card: every line identifies the person",
         sensitive=["Arjun Kapoor", "+44 20 7946 0958", "arjun.kapoor@northwind.co.uk", "12 King's Road", "SW3 4RP"])

    im, d = card((860, 540), (25, 45, 95))
    d.rounded_rectangle((60, 140, 180, 220), 10, fill=(220, 190, 90))
    d.text((60, 280), "5423 1876 0091 2433", fill=WHITE, font=font(56))
    d.text((60, 380), "VALID THRU 11/29", fill=WHITE, font=font(30))
    d.text((60, 440), "DAVID MOREAU", fill=WHITE, font=font(38))
    im.save(OUT / "I20_credit_card.png")
    case("I20", "I20_credit_card.png", "full", "document", "payment card photo: Luhn-valid number, expiry, holder name",
         sensitive=["5423 1876 0091 2433", "11/29", "DAVID MOREAU"])

    im, d = card((900, 640))
    d.rectangle((10, 10, 890, 630), outline=BLACK, width=5)
    text_lines(d, ["SHIP TO:", "Grace Thompson", "88 Willow Creek Road", "Apt 5C", "Portland, OR 97205",
                   "Phone: (503) 555-0117", "Tracking: 1Z999AA10123456784"], (50, 40), 36)
    im.save(OUT / "I21_shipping_label.png")
    case("I21", "I21_shipping_label.png", "full", "document", "shipping label with name, multi-line address, phone",
         sensitive=["Grace Thompson", "88 Willow Creek Road", "97205", "(503) 555-0117"], keep=["SHIP TO:"])

    im, d = card((1150, 300), (40, 42, 54))
    text_lines(d, ["AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE",
                   "AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
                   "DB_PASSWORD=hunter2!Secure"], (30, 50), 30, (240, 240, 210))
    im.save(OUT / "I22_env_file.png")
    case("I22", "I22_env_file.png", "full", "screenshot", "screenshot of a .env file full of credentials",
         sensitive=["AKIAIOSFODNN7EXAMPLE", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", "hunter2!Secure"],
         keep=["AWS_ACCESS_KEY_ID", "DB_PASSWORD"])

    im, d = card((1000, 700), (245, 245, 240))
    text_lines(d, ["First National Bank", "", "Dear Mr. Haruto Sato,", "Your account 0045 8812 9931 is overdrawn.",
                   "Statement date: 3 October 2026", "Branch: 19 Queen Street, Auckland 1010"], (50, 50), 34)
    im = im.convert("L").filter(ImageFilter.GaussianBlur(0.6)).convert("RGB")
    im.save(OUT / "I23_scanned_letter.jpg", quality=80,
            exif=exif(make="Epson", model="DS-530", software="ScanSmart 3.1", gps=(36, 50, 174, 46)))
    case("I23", "I23_scanned_letter.jpg", "full", "document",
         "grayscale, slightly blurred scan with JPEG artefacts, plus scanner EXIF and GPS",
         sensitive=["Haruto Sato", "0045 8812 9931", "19 Queen Street"],
         metadata=["GPS location", "Make", "Model", "Software"])

    im, d = card((1100, 640), (230, 225, 240))
    d.text((30, 20), "PASSPORT", fill=(80, 40, 120), font=font(40))
    f = place_face(im, (280, 320), (40, 90), crop=(110, 10, 340, 275), gray=True)
    text_lines(d, ["Surname: MARKOVA", "Given names: ELENA", "Nationality: BRITISH CITIZEN",
                   "Passport No: 548213097"], (360, 110), 32)
    d.text((40, 520), "P<GBRMARKOVA<<ELENA<<<<<<<<<<<<<<<<<<<<<<<<<", fill=BLACK, font=font(30))
    d.text((40, 570), "5482130974GBR8803121F3001019<<<<<<<<<<<<<<06", fill=BLACK, font=font(30))
    im.save(OUT / "I24_passport.png")
    case("I24", "I24_passport.png", "full", "document",
         "passport page: grayscale photo, name fields and machine-readable zone lines",
         faces=[f], sensitive=["MARKOVA", "ELENA", "548213097", "P<GBRMARKOVA<<ELENA"])

    im, d = card((1260, 560), (250, 250, 250))
    boxes = []
    for i, name in enumerate(["Dr. Anika Rao", "Victor Chen", "Lena Novak"]):
        x = 30 + i * 410
        boxes.append(place_face(im, (380, 380), (x, 30)))
        d.text((x + 40, 450), name, fill=BLACK, font=font(36))
    im.save(OUT / "I25_team_photo.jpg", quality=95,
            exif=exif(make="Apple", model="iPhone 15", artist="Anika Rao", gps=(52, 31, 13, 24)))
    case("I25", "I25_team_photo.jpg", "full", "photo", "three faces with name captions and GPS EXIF",
         faces=boxes, sensitive=["Anika Rao", "Victor Chen", "Lena Novak"],
         metadata=["GPS location", "Make", "Model", "Artist"])

    with open(HERE / "image_cases.jsonl", "w", encoding="utf-8", newline="\n") as fh:
        for row in CASES:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(CASES)} images to {OUT}")


if __name__ == "__main__":
    build()
