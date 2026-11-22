import pytest

from sanitizer.pii import find_pii, luhn_valid, iban_valid, verhoeff_valid


def kinds(text, **kw):
    return [(s.kind, text[s.start:s.end]) for s in find_pii(text, **kw)]


def test_validators():
    assert luhn_valid("4111 1111 1111 1111")
    assert not luhn_valid("4111 1111 1111 1112")
    assert iban_valid("GB82 WEST 1234 5698 7654 32")
    assert not iban_valid("GB82 WEST 1234 5698 7654 33")
    assert verhoeff_valid("2363")        # Wikipedia's worked example: 236 -> check digit 3
    assert not verhoeff_valid("2364")


def test_does_not_return_overlapping_spans():
    assert [k for k, _ in kinds("Contact a.b@example.com or +91 98765 43210")] == ["EMAIL", "PHONE"]


@pytest.mark.parametrize("text,expected", [
    ("Card: 4111 1111 1111 1111 now.", ("CARD_NUMBER", "4111 1111 1111 1111")),
    ("key sk_test_abcdefghijklmnopqrstuvwxyz123456 leaked", ("API_KEY", "sk_test_abcdefghijklmnopqrstuvwxyz123456")),
    ("AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE", ("API_KEY", "AKIAIOSFODNN7EXAMPLE")),
    ("server 192.168.1.25 is down", ("IP_ADDRESS", "192.168.1.25")),
    ("mac 3C:22:FB:9A:1B:7E blocked", ("MAC_ADDRESS", "3C:22:FB:9A:1B:7E")),
    ("My SSN is 123-45-6789.", ("SSN", "123-45-6789")),
    ("IBAN GB82 WEST 1234 5698 7654 32 failed", ("IBAN", "GB82 WEST 1234 5698 7654 32")),
    ("born on July 2nd", ("DATE", "July 2nd")),
    ("lives at 42 Oak Street, Seattle. Please call.", ("ADDRESS", "42 Oak Street, Seattle")),
    ("my password is hunter2.", ("PASSWORD", "hunter2")),
    ("patient ID: MRN-48213 admitted", ("ID_NUMBER", "MRN-48213")),
    ("My name is Priya Nair and", ("PERSON", "Priya Nair")),
    ("Contact Dr. Emily Chen today", ("PERSON", "Emily Chen")),
    ("I live in Bengaluru and", ("LOCATION", "Bengaluru")),
])
def test_recognizers(text, expected):
    assert expected in kinds(text)


def test_quoted_dict_keys_are_recognized():
    text = "db_config = {'host': 'prod-db.internal.acme.io', 'password': 'Tr0ub4dor&3'}"
    found = kinds(text)
    assert ("HOSTNAME", "prod-db.internal.acme.io") in found
    assert ("PASSWORD", "Tr0ub4dor&3") in found


def test_credentials_in_code_hide_values_only():
    text = "psycopg2.connect(host=114.165.14.1, database=production, user=Estrella, password=>@]@x5<pSA)"
    found = kinds(text)
    assert ("IP_ADDRESS", "114.165.14.1") in found
    assert ("USERNAME", "Estrella") in found
    assert ("PASSWORD", ">@]@x5<pSA") in found
    assert not any(v in ("psycopg2.connect", "production") for _, v in found)


@pytest.mark.parametrize("text", [
    "I take 500 mg twice daily for 10 days.",
    "Python 3.10.12 is installed.",
    "User: Hi there, you may 2 things",
    "I have had a headache and sore throat for two days.",
    "The meeting has 25 people and costs 1200 dollars.",
])
def test_no_false_positives(text):
    assert kinds(text) == []


def test_entity_propagation_hides_every_mention():
    text = "My name is Rahul Sharma. Rahul lives in Pune and Sharma works there."
    found = kinds(text)
    assert ("PERSON", "Rahul Sharma") in found
    assert ("PERSON", "Rahul") in found
    assert ("PERSON", "Sharma") in found
    assert ("LOCATION", "Pune") in found


def test_middle_initial_and_extended_contact_fields():
    assert ("PERSON", "Anita Q. Baker") in kinds("Signed by Anita Q. Baker.")
    assert ("POSTAL_CODE", "12345-678") in kinds("ZIP: 12345-678")
    assert ("PHONE", "(898)666-3621x0135") in kinds("Desk: (898)666-3621x0135")


def test_existing_masks_are_separate_from_new_edits(monkeypatch):
    from sanitizer.sanitizer import PromptSanitizer

    monkeypatch.setattr(PromptSanitizer, "_load_prosan_components", lambda self: None)
    sanitizer = PromptSanitizer()
    sanitizer.config.ner_backend = "none"
    sanitizer.config.surrogate_style = "placeholder"
    candidates = []
    def inspect(prompt, words, *args):
        candidates.extend(w["word"] for w in words)
        return 0.0, 0.0
    monkeypatch.setattr(sanitizer, "_contextual", inspect)
    text = "Keep <EMAIL_1> or [REDACTED]; password=****; email: alice@example.org"
    result = sanitizer.sanitize(text, evaluate=False)
    assert result.text == text.replace("alice@example.org", "<EMAIL_2>")
    assert [s["word"] for s in result.already_masked] == ["<EMAIL_1>", "[REDACTED]", "****"]
    assert all(text[s["start"]:s["end"]] == s["word"] for s in result.already_masked)
    assert "REDACTED" not in candidates and "EMAIL_1" not in candidates
    assert sanitizer.restore(result.text, result.mapping) == text
    assert not sanitizer.sanitize(result.text, evaluate=False).selected_words
    assert ("PASSWORD", "abc***") in kinds("password=abc***")


def test_private_profile_and_employer_context():
    assert ("TITLE", "Dr.") in kinds("Ask Dr. Amara Shah.")
    assert ("TITLE", "Systems engineer") in kinds("Systems engineer\n\nContact details:")
    assert ("TITLE", "Operations manager") in kinds("Occupation: Operations manager")
    assert ("ORGANIZATION", "Cedar Labs") in kinds("I worked for Cedar Labs as an engineer.")
    assert not kinds("The engineer explained how relays work for a controller.")
    assert not kinds("Turn onto River Dr.")
    text = "Update the billing address with 42 Cedar Road for this card: 4111111111111111"
    assert ("ADDRESS", "42 Cedar Road") in kinds(text)
    assert ("CARD_NUMBER", "4111111111111111") in kinds(text)


def test_post_office_box_and_address_block_offsets():
    assert ("ADDRESS", "P.O. Box 821") in kinds("Mail to P.O. Box 821.")
    for separator in ("\n", "\r\n"):
        record = separator.join(["314 Cedarstrasse 26", "London", "12345"])
        assert ("ADDRESS", record) in kinds(">" + record + separator + "Phone: +1 555 123 4567")


def test_explicit_names_and_delivery_points():
    assert ("PERSON", "ines alves") in kinds("My name is ines alves and I need help.")
    assert ("PERSON", "Kowalski") in kinds("Her maiden name is Kowalski.")
    assert not kinds("My name is not available.")
    assert ("ADDRESS", "ul. Lipowa 16") in kinds("Meet me at ul. Lipowa 16.")
    military = "CMR 122 Box 841\nAPO AE 09012"
    assert ("ADDRESS", military) in kinds(military)
    assert ("ADDRESS", "418 Cedarstrand 12\nSuite 4\nOslo") in kinds(
        "418 Cedarstrand 12\nSuite 4\nOslo\nPhone: +1 202 555 0198"
    )


def test_international_address_boundaries():
    assert ("ADDRESS", "Lindenstraat 84") in kinds("Meet at Lindenstraat 84 before lunch.")
    assert ("ADDRESS", "Via delle Rose 17") in kinds("Walk past Via delle Rose 17 today.")
    for newline in ("\n", "\r\n"):
        record = newline.join(["415 26 Orchard Street", "Unit 8", "North Bay", "Canada"])
        for following in ("Reference: A", "Please ring twice.", "Leave at reception.", "We will arrive tomorrow."):
            assert ("ADDRESS", record) in kinds(record + newline + following)
    assert not kinds("The release candidate passed gate 12.")

    text = "Meet at Lindenstraat 84 or Lindenstraat 92. Contact: <PERSON_1>"
    assert [v for k, v in kinds(text) if k == "ADDRESS"] == ["Lindenstraat 84", "Lindenstraat 92"]
    assert all("<PERSON_1>" not in v for _, v in kinds(text))


# ── Multi-word entities are captured whole (connecting words included) ──

def test_firm_name_list_is_one_organization():
    assert ("ORGANIZATION", "Hawkins, Richardson and Santana") in kinds("Tomomi is from Hawkins, Richardson and Santana")


def test_day_list_is_not_a_firm():
    assert not any(k == "ORGANIZATION" for k, _ in kinds("We meet Monday, Tuesday and Friday."))


def test_street_corner_is_one_address():
    found = kinds("What is your address? it is the corner of Dalbraut 99 and Frederikke Coves")
    assert ("ADDRESS", "the corner of Dalbraut 99 and Frederikke Coves") in found


def test_template_placeholders_are_never_pii():
    assert kinds("The corner of {street_name} and {street_name}") == []


def test_ship_military_address():
    assert ("ADDRESS", "USNS Bergman\nFPO AP 93757") in kinds("Send it to USNS Bergman\nFPO AP 93757 please")


def test_house_number_and_postal_lines_join_the_address():
    found = kinds("I'd like a taxi to 692 Kuuse 53 Apt. 641, Grubbenvorst, Netherlands 04020. Thanks")
    assert any(k == "ADDRESS" and v.startswith("692 ") and v.endswith("04020") for k, v in found)


def test_prefix_street_with_number_first():
    assert ("ADDRESS", "33 avenue de Provence") in kinds("Verda lives on 33 avenue de Provence street.")


# ── Self-described affiliation vs. adjectives about things ──

def test_self_described_nationality_is_nrp():
    assert ("NRP", "cambodian") in kinds("I am cambodian but I live in United Kingdom.")


def test_nationality_about_a_thing_is_not_flagged():
    assert not any(k == "NRP" for k, _ in kinds("The Russian government began the process."))


# ── Precision fixes ──

def test_card_number_is_not_matched_inside_an_identifier():
    assert not any(k == "CARD_NUMBER" for k, _ in kinds("My license number is U62928788557186"))


def test_phone_includes_international_prefix():
    assert any(k == "PHONE" and v.startswith("+1") for k, v in kinds("Desk: +1-903-140-4508x769"))


def test_surname_prefixes_are_not_code_identifiers():
    from sanitizer.pii import _is_code_like

    assert not _is_code_like("McDowell") and not _is_code_like("MacLeod")
    assert _is_code_like("ImportError")


def test_weekday_after_temporal_cue_is_a_date_not_a_person():
    found = kinds("We'll meet Monday at the office")
    assert ("DATE", "Monday") in found
    assert not any(k == "PERSON" for k, _ in found)
