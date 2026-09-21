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
    ("lives at 42 Oak Street, Seattle", ("ADDRESS", "42 Oak Street")),
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
