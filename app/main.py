from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from sanitizer import PromptSanitizer, PromptEvaluator

app = FastAPI(
    title="ProSan Prompt Privacy Sanitizer API",
    description="API for utility-preserving prompt privacy sanitization.",
    version="2.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

sanitizer = PromptSanitizer()


class Overrides(BaseModel):
    mode: Optional[str] = Field(None, description="'prosan' or 'pii_only'")
    surrogate_style: Optional[str] = Field(None, description="'realistic' or 'placeholder'")
    lambda_scale: Optional[float] = Field(None, description="Override dynamic protection ratio scale factor lambda")
    min_privacy: Optional[float] = Field(None, description="Override privacy-risk threshold")
    eta: Optional[float] = Field(None, description="Override privacy weight hyperparameter eta")
    tau: Optional[float] = Field(None, description="Override cumulative probability threshold tau")
    sampling_mode: Optional[str] = Field(None, description="Override sampling mode: 'sample' or 'max'")

    def to_overrides(self) -> Dict[str, Any]:
        # Only this base model's own fields are config overrides — a subclass
        # (SanitizeRequest, BatchRequest) adds request fields like "prompt"
        # that must never be forwarded to SanitizerConfig.
        return {k: v for k, v in self.model_dump(include=set(Overrides.model_fields)).items() if v is not None}


class SanitizeRequest(Overrides):
    prompt: str = Field(..., description="Prompt text to desensitize")
    history: Optional[List[str]] = Field(None, description="Multi-turn context history")
    evaluate: bool = Field(True, description="Compute perplexity and PHR")


class BatchRequest(Overrides):
    prompts: List[str]


class SanitizeResponse(BaseModel):
    original: str
    sanitized: str
    selected_words: List[Dict[str, Any]]
    mapping: Dict[str, str] = {}
    H_q: float
    gamma_q: float
    perplexity: Optional[float] = None
    original_perplexity: Optional[float] = None
    phr: Optional[float] = None


class RestoreRequest(BaseModel):
    text: str = Field(..., description="LLM response written against the sanitized prompt")
    mapping: Dict[str, str] = Field(..., description="The mapping returned by /sanitize")


class EvaluateRequest(BaseModel):
    original: str
    sanitized: str
    sensitive_items: Optional[List[str]] = Field(None, description="List of target sensitive keywords")


class EvaluateResponse(BaseModel):
    original_perplexity: float
    sanitized_perplexity: float
    phr_metrics: Optional[Dict[str, Any]] = None


def _run(prompt: str, overrides: Dict[str, Any], history=None, evaluate=False):
    try:
        return sanitizer.sanitize(prompt, history=history, config_overrides=overrides, evaluate=evaluate)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/health")
def health():
    return {"status": "ok", "service": "ProSan Prompt Sanitizer"}


@app.get("/config")
def get_config():
    return {k: v for k, v in vars(sanitizer.config).items()}


@app.post("/sanitize", response_model=SanitizeResponse)
def sanitize(request: SanitizeRequest):
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    # Overrides apply to this request only; the shared sanitizer is never mutated.
    result = _run(request.prompt, request.to_overrides(), request.history, request.evaluate)
    return {
        "original": request.prompt,
        "sanitized": result.text,
        "selected_words": result.selected_words,
        "mapping": result.mapping,
        "H_q": result.H_q,
        "gamma_q": result.gamma_q,
        "perplexity": result.perplexity,
        "original_perplexity": result.original_perplexity,
        "phr": result.phr,
    }


@app.post("/sanitize/batch")
def sanitize_batch(request: BatchRequest):
    """Original -> sanitized training pairs (the dashboard downloads these as JSONL)."""
    overrides = request.to_overrides()
    rows = []
    for prompt in request.prompts:
        if not prompt.strip():
            continue
        result = _run(prompt, overrides)
        rows.append({"original": prompt, "sanitized": result.text, "selected_words": result.selected_words})
    return rows


@app.post("/restore")
def restore(request: RestoreRequest):
    return {"text": PromptSanitizer.restore(request.text, request.mapping)}


@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate(request: EvaluateRequest):
    ev: PromptEvaluator = sanitizer.evaluator
    orig_ppl, san_ppl = ev.perplexity_batch([request.original, request.sanitized])

    phr_info = None
    if request.sensitive_items:
        phr_info = ev.calculate_phr(request.sensitive_items, request.sanitized)

    return {
        "original_perplexity": orig_ppl,
        "sanitized_perplexity": san_ppl,
        "phr_metrics": phr_info,
    }
