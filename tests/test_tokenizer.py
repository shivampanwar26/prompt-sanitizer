from sanitizer.tokenizer import extract_words


def test_extract_words():
    text = "John lives in Delhi."
    words = extract_words(text)

    assert len(words) == 4
    assert words[0]["word"] == "John"
    assert words[0]["start"] == 0
    assert words[0]["end"] == 4
    assert words[-1]["word"] == "Delhi"


def test_extract_words_with_punctuation():
    text = "My email is test@example.com and phone is 123-456."
    words = extract_words(text)
    word_tokens = [w["word"] for w in words]

    assert "test@example.com" in word_tokens
    assert "123-456" in word_tokens
