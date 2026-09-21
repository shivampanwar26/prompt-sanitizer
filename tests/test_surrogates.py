from datetime import date

from sanitizer.pii import luhn_valid
from sanitizer.sanitizer import PromptSanitizer
from sanitizer.surrogates import SurrogateGenerator


def test_consistent_and_deterministic():
    a, b = SurrogateGenerator(secret="s"), SurrogateGenerator(secret="s")
    assert a.get("PERSON", "John Doe") == a.get("PERSON", "john doe") == b.get("PERSON", "John Doe")
    assert a.get("PERSON", "John Doe") != "John Doe"


def test_name_parts_follow_full_name():
    g = SurrogateGenerator()
    full = g.get("PERSON", "Rahul Sharma")
    assert g.get("PERSON", "Rahul") == full.split()[0]
    assert g.get("PERSON", "Sharma") == full.split()[1]


def test_distinct_originals_get_distinct_surrogates():
    g = SurrogateGenerator()
    names = [g.get("PERSON", n) for n in ["Ann", "Bob", "Cid", "Dee", "Eve", "Fay"]]
    assert len(set(names)) == len(names)


def test_format_preserving_values():
    g = SurrogateGenerator()
    phone = g.get("PHONE", "+91 98765 43210")
    assert phone.startswith("+91 ") and len(phone) == len("+91 98765 43210")
    card = g.get("CARD_NUMBER", "4111 1111 1111 1111")
    assert len(card) == 19 and not luhn_valid(card)
    assert g.get("IP_ADDRESS", "10.0.4.17").startswith("192.0.2.")


def test_date_shift_preserves_intervals():
    g = SurrogateGenerator()
    d1, d2 = g.get("DATE", "2024-01-10"), g.get("DATE", "2024-03-01")
    assert d1 != "2024-01-10"
    gap = date.fromisoformat(d2) - date.fromisoformat(d1)
    assert gap.days == (date(2024, 3, 1) - date(2024, 1, 10)).days
    assert g.get("DATE", "July 2nd").split()[1][-2:] in ("st", "nd", "rd", "th")


def test_placeholder_style_and_restore():
    g = SurrogateGenerator(style="placeholder")
    assert g.get("EMAIL", "a@b.com") == "<EMAIL_1>"
    assert g.get("EMAIL", "c@d.com") == "<EMAIL_2>"
    restored = PromptSanitizer.restore("Reply to <EMAIL_1> and <EMAIL_2>.", g.mapping)
    assert restored == "Reply to a@b.com and c@d.com."


def test_restore_realistic_names_is_word_bounded():
    mapping = {"Alex": "John", "Alex Reed": "John Doe"}
    assert PromptSanitizer.restore("Alex Reed met Alex at Alexandria.", mapping) == "John Doe met John at Alexandria."
