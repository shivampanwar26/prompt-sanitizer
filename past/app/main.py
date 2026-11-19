from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from sanitizer import PromptSanitizer, SanitizerConfig, PromptEvaluator

app = FastAPI(
    title="ProSan Prompt Privacy Sanitizer API",
    description="API for utility-preserving prompt privacy sanitization.",
    version="1.0.0",
)

sanitizer = PromptSanitizer()


class SanitizeRequest(BaseModel):
    prompt: str = Field(..., description="Prompt text to desensitize")
    history: Optional[List[str]] = Field(None, description="Multi-turn context history")
    lambda_scale: Optional[float] = Field(None, description="Override dynamic protection ratio scale factor lambda")
    eta: Optional[float] = Field(None, description="Override privacy weight hyperparameter eta")
    tau: Optional[float] = Field(None, description="Override cumulative probability threshold tau")
    sampling_mode: Optional[str] = Field(None, description="Override sampling mode: 'sample' or 'max'")


class SanitizeResponse(BaseModel):
    original: str
    sanitized: str
    selected_words: List[Dict[str, Any]]
    H_q: float
    gamma_q: float
    perplexity: Optional[float] = None
    original_perplexity: Optional[float] = None
    phr: Optional[float] = None


class EvaluateRequest(BaseModel):
    original: str
    sanitized: str
    sensitive_items: Optional[List[str]] = Field(None, description="List of target sensitive keywords")


class EvaluateResponse(BaseModel):
    original_perplexity: float
    sanitized_perplexity: float
    phr_metrics: Optional[Dict[str, Any]] = None


@app.get("/health")
def health():
    return {"status": "ok", "service": "ProSan Prompt Sanitizer"}


@app.get("/config")
def get_config():
    cfg = sanitizer.config
    return {
        "causal_model": cfg.causal_model,
        "mask_model": cfg.mask_model,
        "lambda_scale": cfg.lambda_scale,
        "min_privacy": cfg.min_privacy,
        "max_importance": cfg.max_importance,
        "eta": cfg.eta,
        "tau": cfg.tau,
        "top_k": cfg.top_k,
        "sampling_mode": cfg.sampling_mode,
        "pos_filter_enabled": cfg.pos_filter_enabled,
        "device": cfg.device,
    }


@app.post("/sanitize", response_model=SanitizeResponse)
def sanitize(request: SanitizeRequest):
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    if request.lambda_scale is not None:
        sanitizer.selector.lambda_scale = request.lambda_scale
    if request.eta is not None:
        sanitizer.replacement.eta = request.eta
    if request.tau is not None:
        sanitizer.replacement.tau = request.tau
    if request.sampling_mode is not None:
        sanitizer.replacement.sampling_mode = request.sampling_mode

    result = sanitizer.sanitize(
        prompt=request.prompt,
        history=request.history,
    )

    return {
        "original": request.prompt,
        "sanitized": result.text,
        "selected_words": result.selected_words,
        "H_q": result.H_q,
        "gamma_q": result.gamma_q,
        "perplexity": result.perplexity,
        "original_perplexity": result.original_perplexity,
        "phr": result.phr,
    }


@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate(request: EvaluateRequest):
    ev = sanitizer.evaluator or PromptEvaluator(model_name=sanitizer.config.causal_model, device=sanitizer.config.device)
    orig_ppl = ev.perplexity(request.original)
    san_ppl = ev.perplexity(request.sanitized)

    phr_info = None
    if request.sensitive_items:
        phr_info = ev.calculate_phr(request.sensitive_items, request.sanitized)

    return {
        "original_perplexity": orig_ppl,
        "sanitized_perplexity": san_ppl,
        "phr_metrics": phr_info,
    }
