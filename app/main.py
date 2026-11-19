from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sanitizer import PromptSanitizer, PromptEvaluator

app = FastAPI(
    title="Prompt Privacy Sanitizer API",
    description="API for utility-preserving prompt privacy sanitization."
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

sanitizer = PromptSanitizer()


class SanitizeRequest(BaseModel):
    prompt: str
    evaluate: bool=False


class SanitizeResponse(BaseModel):
    original: str
    sanitized: str
    selected_words: List[dict]
    mapping: dict
    H_q: float
    gamma_q: float
    perplexity: float|None = None
    original_perplexity: float|None = None
    phr: float|None = None


@app.get("/health")
def health():
    return {"status": "ok", "service": "ProSan Prompt Sanitizer"}


@app.post("/sanitize", response_model=SanitizeResponse)
def sanitize(request: SanitizeRequest):
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    result = sanitizer.sanitize(request.prompt, evaluate=request.evaluate)
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