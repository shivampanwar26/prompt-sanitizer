import pytest

from sanitizer.config import SanitizerConfig
from sanitizer.evaluator import PromptEvaluator
from sanitizer.privacy import PrivacyCalculator, privacy_prior, is_sentence_initial
from sanitizer.sanitizer import PromptSanitizer
from sanitizer.selector import SelectionUnit, WordSelector
from sanitizer.similarity import SimilarityCalculator


# ── fast unit tests (no models) ────────────────────────────────

def test_config_overrides_are_typed_and_validated():
    cfg = SanitizerConfig().with_overrides({"eta": "2", "pos_filter_enabled": "false"})
    assert cfg.eta == 2.0 and cfg.pos_filter_enabled is False
    with pytest.raises(ValueError):
        SanitizerConfig().with_overrides({"nope": 1})


def test_fallback_similarity():
    sim = SimilarityCalculator()
    assert sim.similarity("doctor", "doctor") == 1.0
    assert sim.fallback_similarity("John", "Johnathan") > 0.5


def test_privacy_prior_prefers_identities_over_dictionary_words():
    assert privacy_prior("Estrella", "NNP", sentence_initial=False) > privacy_prior("fever", "NN", False)
    assert privacy_prior("MRN4829", "NN", False) > privacy_prior("2", "CD", False)
    assert is_sentence_initial("Hello. Jack", 7) and not is_sentence_initial("I met Jack", 6)


def test_privacy_prior_ignores_sentence_initial_mistagging():
    """A tagger quirk ("Translate" -> NNP at sentence start) must not outweigh
    the word being ordinary English vocabulary."""
    assert privacy_prior("Translate", "NNP", sentence_initial=True) < 0.5
    assert privacy_prior("Estrella", "NNP", sentence_initial=True) >= 0.5


def test_privacy_prior_spares_versioned_library_names():
    for name in ("boto3", "gpt4", "oauth2", "python3"):
        assert privacy_prior(name, "NN", False) < 0.5
    assert privacy_prior("MRN4829", "NN", False) >= 0.5   # a real ID is still flagged


def test_privacy_is_absolute_and_max_pooled():
    words = [{"word": "Estrella", "start": 0, "end": 8}, {"word": "estrella", "start": 20, "end": 28}]
    risk, raw = PrivacyCalculator(saturation=8).calculate("Estrella said hi to estrella", words, [14.0, 2.0], ["NNP", "NN"])
    assert risk[0] == risk[1] > 0.5          # later mention inherits the first mention's risk
    assert raw == [14.0, 14.0]


def test_selector_threshold_budget_and_entity_units():
    selector = WordSelector(lambda_scale=0.5, min_privacy=0.5, max_importance=0.6)
    text = "Estrella met Garcia and Estrella left"
    words = [{"word": w, "start": text.index(w, s), "end": text.index(w, s) + len(w)}
             for w, s in [("Estrella", 0), ("met", 0), ("Garcia", 0), ("and", 0), ("Estrella", 10), ("left", 0)]]
    units, H_q, gamma_q = selector.select(
        words, importance=[0.1, 0.9, 0.2, 0.1, 0.1, 0.8], privacy=[0.9, 0.1, 0.8, 0.0, 0.9, 0.2],
        raw_privacy=[12, 3, 11, 1, 12, 4], pos_tags=["NNP", "VBD", "NNP", "CC", "NNP", "VBD"], full_text=text,
    )
    assert 0 < gamma_q <= 0.5
    by_text = {u.text: u for u in units}
    assert set(by_text) == {"Estrella", "Garcia"}
    assert len(by_text["Estrella"].occurrences) == 2   # every mention is selected


def test_selector_does_not_treat_sentence_initial_caps_as_identity():
    """"Translate this" must not get the same free pass as "Estrella said this"."""
    selector = WordSelector(max_importance=0.5)
    text = "Translate this sentence."
    words = [{"word": "Translate", "start": 0, "end": 9}, {"word": "this", "start": 10, "end": 14},
             {"word": "sentence", "start": 15, "end": 23}]
    # High importance (it is the main verb) and, unrealistically, high privacy too:
    # a sentence-initial capital alone must not be enough to override protection.
    units, _, _ = selector.select(words, importance=[0.9, 0.1, 0.2], privacy=[0.9, 0.1, 0.3],
                                  raw_privacy=[10, 1, 2], pos_tags=["VB", "DT", "NN"], full_text=text)
    assert "Translate" not in {u.text for u in units}


def test_selector_leaves_benign_prompt_alone():
    selector = WordSelector()
    words = [{"word": "fever", "start": 9, "end": 14}]
    units, _, _ = selector.select(words, [0.2], [0.3], [11.0], ["NN"], full_text="I have a fever")
    assert units == []


def test_choose_falls_back_when_scores_underflow_to_zero():
    """Degenerate scores (e.g. a zero embedding vector) must never crash sampling."""
    from sanitizer.replacement import ReplacementGenerator

    gen = object.__new__(ReplacementGenerator)  # skip loading the masked LM
    gen.eta, gen.sampling_mode, gen.seed = 1.0, "sample", 42
    gen.similarity_calc = type("Sim", (), {"similarity_to_ids": staticmethod(lambda w, ids: [0.0] * len(ids))})()
    unit = SelectionUnit(text="x", occurrences=[(0, 1)], importance=0.5, privacy=0.5, raw_privacy=1.0, pos_tag="NN")

    assert gen.choose(unit, [("a", 1, 0.0), ("b", 2, 0.0)]) in ("a", "b")


def test_phr_uses_whole_words():
    assert PromptEvaluator.calculate_phr(["Sam"], "Samsung phones")["phr"] == 100.0
    assert PromptEvaluator.calculate_phr(["John", "Delhi"], "John lives in Delhi")["phr"] == 0.0


def pii_only(**kw):
    return PromptSanitizer(SanitizerConfig(mode="pii_only", ner_backend="none", **kw))


def test_pii_only_mode_loads_no_models_and_keeps_medical_text():
    sanitizer = pii_only(surrogate_style="placeholder")
    assert sanitizer._analyzer is None
    result = sanitizer.sanitize("My email is shivam@gmail.com; I have a fever.", evaluate=False)
    assert result.text == "My email is <EMAIL_1>; I have a fever."
    assert result.selected_words[0]["pos_tag"] == "PII:EMAIL"
    assert PromptSanitizer.restore(result.text, result.mapping) == "My email is shivam@gmail.com; I have a fever."


def test_pii_keeps_sentence_punctuation():
    result = pii_only(surrogate_style="placeholder").sanitize(
        "Email patient.records@example.org. Card: 4111 1111 1111 1111 now.", evaluate=False)
    assert result.text == "Email <EMAIL_1>. Card: <CARD_NUMBER_1> now."


def test_session_keeps_entities_consistent_across_turns():
    sanitizer = pii_only()
    session = sanitizer.new_session()
    first = sanitizer.sanitize("My name is Rahul Sharma.", evaluate=False, session=session)
    second = sanitizer.sanitize("Is Rahul eligible for the scheme?", evaluate=False, session=session)
    fake_first_name = first.text.split("is ")[1].split()[0]
    assert "Rahul" not in second.text
    assert fake_first_name in second.text


# ── model-backed tests ─────────────────────────────────────────

@pytest.fixture(scope="module")
def full_sanitizer():
    return PromptSanitizer(SanitizerConfig(device="cpu", sampling_mode="max"))


@pytest.mark.models
def test_prompt_evaluator():
    evaluator = PromptEvaluator(model_name="distilgpt2", device="cpu")
    assert evaluator.perplexity("This is a simple test sentence.") > 1.0


@pytest.mark.models
def test_full_pipeline_hides_identities_and_keeps_task(full_sanitizer):
    prompt = "Jack was born on July 2nd and has had a fever of 103°F for 2 days. What steps should be taken?"
    result = full_sanitizer.sanitize(prompt)
    assert "Jack" not in result.text and "July 2nd" not in result.text
    assert "fever" in result.text and "103" in result.text
    assert result.phr == 100.0 and result.perplexity is not None


@pytest.mark.models
def test_full_pipeline_leaves_benign_prompt_unchanged(full_sanitizer):
    prompt = "What are safe home-care steps for a mild fever?"
    assert full_sanitizer.sanitize(prompt, evaluate=False).text == prompt


@pytest.mark.models
def test_multi_turn_history(full_sanitizer):
    history = ["What is the patient's condition?", "The patient is 45 years old."]
    result = full_sanitizer.sanitize("Patient John Doe has dyspnea.", history=history, evaluate=False)
    assert "John" not in result.text and "dyspnea" in result.text


@pytest.mark.models
def test_fastapi_endpoints():
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    assert "causal_model" in client.get("/config").json()

    res = client.post("/sanitize", json={"prompt": "My name is Alice and my email is alice@corp.com."})
    assert res.status_code == 200
    data = res.json()
    assert "alice@corp.com" not in data["sanitized"] and data["perplexity"] is not None

    restored = client.post("/restore", json={"text": data["sanitized"], "mapping": data["mapping"]}).json()
    assert "alice@corp.com" in restored["text"]

    rows = client.post("/sanitize/batch", json={"prompts": ["Email me@example.com.", "I have a fever."], "mode": "pii_only"}).json()
    assert rows[1]["sanitized"] == "I have a fever."
    assert client.post("/sanitize", json={"prompt": "hi", "mode": "bogus"}).status_code == 400
