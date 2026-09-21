import pytest
from fastapi.testclient import TestClient

from sanitizer.config import SanitizerConfig
from sanitizer.similarity import SimilarityCalculator
from sanitizer.selector import WordSelector
from sanitizer.privacy import normalize, PrivacyCalculator
from sanitizer.evaluator import PromptEvaluator
from sanitizer.sanitizer import PromptSanitizer
from sanitizer.pii import find_pii
from app.main import app


def test_similarity_calculator():
    sim_calc = SimilarityCalculator()
    # Identical words
    assert sim_calc.similarity("doctor", "doctor") == 1.0
    # Similar words
    sim_wn = sim_calc.similarity("doctor", "physician")
    assert sim_wn > 0.3
    # Fallback
    sim_fall = sim_calc.fallback_similarity("John", "Johnathan")
    assert sim_fall > 0.5


def test_word_selector_dynamic_ratio():
    selector = WordSelector(lambda_scale=0.5, min_privacy=0.3, pos_filter_enabled=True)
    words = [
        {"word": "John", "start": 0, "end": 4},
        {"word": "lives", "start": 5, "end": 10},
        {"word": "in", "start": 11, "end": 13},
        {"word": "Delhi", "start": 14, "end": 19},
    ]
    importance = [0.1, 0.8, 0.2, 0.1]
    privacy = [0.9, 0.2, 0.1, 0.85]
    raw_privacy = [8.5, 2.0, 1.0, 8.0]

    selected, H_q, gamma_q = selector.select(words, importance, privacy, raw_privacy)

    assert H_q > 0.0
    assert 0.0 < gamma_q <= 0.5
    assert len(selected) >= 1
    # Check that function words like 'in' (POS IN) are filtered out if POS enabled
    selected_words = [x.word for x in selected]
    assert "in" not in selected_words


def test_repeated_entity_pooling():
    # Privacy scores normalization test
    scores = [10.0, 2.0, 10.0]
    norm = normalize(scores)
    assert norm[0] == norm[2] == 1.0
    assert norm[1] == 0.0


def test_pii_redacts_email_without_changing_medical_text():
    sanitizer = PromptSanitizer()
    prompt = "My email is shivam@gmail.com; I have a fever."
    text, selected = sanitizer._redact_pii_spans(prompt)

    assert text == "My email is <EMAIL>; I have a fever."
    assert selected[0]["pos_tag"] == "PII:EMAIL"
    assert "fever" in text


def test_pii_detector_does_not_return_overlapping_spans():
    spans = find_pii("Contact a.b@example.com or +91 98765 43210")
    assert [span.kind for span in spans] == ["EMAIL", "PHONE"]


def test_pii_detection_keeps_sentence_punctuation_and_spaces():
    sanitizer = PromptSanitizer()
    text, _ = sanitizer._redact_pii_spans("Email patient.records@example.org. Card: 4111 1111 1111 1111 now.")

    assert text == "Email <EMAIL>. Card: <CARD_NUMBER> now."


def test_prompt_evaluator():
    evaluator = PromptEvaluator(model_name="distilgpt2", device="cpu")
    ppl = evaluator.perplexity("This is a simple test sentence.")
    assert ppl > 1.0

    phr = PromptEvaluator.calculate_phr(["John", "Delhi"], "This is a test sentence in India.")
    assert phr["total"] == 2
    assert phr["hidden"] == 2
    assert phr["phr"] == 100.0


def test_prompt_sanitizer_pipeline():
    config = SanitizerConfig(
        causal_model="distilgpt2",
        mask_model="roberta-base",
        lambda_scale=0.5,
        min_privacy=0.2,
        device="cpu",
        sampling_mode="max",
        seed=42,
    )
    sanitizer = PromptSanitizer(config=config)

    prompt = "My name is John and I live in Delhi. I have severe chest pain."
    result = sanitizer.sanitize(prompt)

    assert isinstance(result.text, str)
    assert result.H_q >= 0.0
    assert result.gamma_q >= 0.0


def test_multi_turn_history():
    config = SanitizerConfig(
        causal_model="distilgpt2",
        mask_model="roberta-base",
        device="cpu",
        sampling_mode="max",
    )
    sanitizer = PromptSanitizer(config=config)

    history = ["What is the patient's condition?", "The patient is 45 years old."]
    prompt = "Patient John Doe has dyspnea."

    result = sanitizer.sanitize(prompt, history=history)
    assert isinstance(result.text, str)


def test_fastapi_endpoints():
    client = TestClient(app)

    # Health check
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"

    # Config
    res_cfg = client.get("/config")
    assert res_cfg.status_code == 200
    assert "causal_model" in res_cfg.json()

    # Sanitize
    res_san = client.post(
        "/sanitize",
        json={
            "prompt": "My name is Alice and I work at Google.",
            "evaluate_perplexity": True,
        },
    )
    assert res_san.status_code == 200
    data = res_san.json()
    assert "sanitized" in data
    assert "H_q" in data
    assert "gamma_q" in data
    assert data["perplexity"] is not None
