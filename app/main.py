import io
import json
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image
from pydantic import BaseModel, Field

from sanitizer import PromptSanitizer, PromptEvaluator, ImageSanitizer

app = FastAPI(
    title="Prompt Privacy Sanitizer API",
    description="API for utility-preserving prompt privacy sanitization."
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
    # The image endpoint returns its report in a custom header; browsers hide
    # custom headers from cross-origin pages unless they are exposed here.
    expose_headers=["X-Sanitization-Report"],
)

sanitizer = PromptSanitizer()


class Overrides(BaseModel):
    """Optional per-request config overrides; never mutate the shared sanitizer."""
    mode: Optional[str] = Field(None, description="'prosan' or 'pii_only'")
    surrogate_style: Optional[str] = Field(None, description="'realistic' or 'placeholder'")
    lambda_scale: Optional[float] = None
    min_privacy: Optional[float] = None
    eta: Optional[float] = None
    tau: Optional[float] = None
    sampling_mode: Optional[str] = Field(None, description="'sample' or 'max'")

    def to_overrides(self) -> Dict[str, Any]:
        # Only this base model's own fields are config overrides; subclasses add
        # request fields such as "prompt" that must not reach SanitizerConfig.
        return {k: v for k, v in self.model_dump(include=set(Overrides.model_fields)).items() if v is not None}


class SanitizeRequest(Overrides):
    prompt: str
    evaluate: bool = False
    history: Optional[List[str]] = None


class BatchRequest(Overrides):
    prompts: List[str]


class SanitizeResponse(BaseModel):
    original: str
    sanitized: str
    selected_words: List[dict]
    already_masked: List[dict]
    mapping: dict
    H_q: float
    gamma_q: float
    perplexity: float|None = None
    original_perplexity: float|None = None
    phr: float|None = None


class RestoreRequest(BaseModel):
    text: str = Field(..., description="LLM response written against the sanitized prompt")
    mapping: Dict[str, str] = Field(..., description="The mapping returned by /sanitize")


class EvaluateRequest(BaseModel):
    original: str
    sanitized: str
    sensitive_items: Optional[List[str]] = None


def _run(prompt: str, overrides: Dict[str, Any], history=None, evaluate: bool = False):
    try:
        return sanitizer.sanitize(prompt, history=history, evaluate=evaluate,
                                  config_overrides=overrides or None)
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

    result = _run(request.prompt, request.to_overrides(), request.history, request.evaluate)
    return {
        "original": request.prompt,
        "sanitized": result.text,
        "selected_words": result.selected_words,
        "mapping": result.mapping,
        "already_masked": result.already_masked,
        "H_q": result.H_q,
        "gamma_q": result.gamma_q,
        "perplexity": result.perplexity,
        "original_perplexity": result.original_perplexity,
        "phr": result.phr,
    }


@app.post("/sanitize/batch")
def sanitize_batch(request: BatchRequest):
    """Original -> sanitized pairs, e.g. for building a training set via the API."""
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


@app.post("/evaluate")
def evaluate(request: EvaluateRequest):
    ev: PromptEvaluator = sanitizer.evaluator
    orig_ppl, san_ppl = ev.perplexity_batch([request.original, request.sanitized])
    phr_info = ev.calculate_phr(request.sensitive_items, request.sanitized) if request.sensitive_items else None
    return {"original_perplexity": orig_ppl, "sanitized_perplexity": san_ppl, "phr_metrics": phr_info}


@app.post("/sanitize/image")
async def sanitize_image(
    file: UploadFile = File(...),
    redact_faces: bool = Query(True),
    redact_text: bool = Query(True),
    deep: bool = Query(False, description="Also run OCR'd text through the full contextual pipeline"),
):
    """Redact faces, sensitive in-image text and metadata; returns the sanitized PNG.

    The sanitization report is returned as JSON in the ``X-Sanitization-Report``
    header, since the response body itself is the image.
    """
    try:
        image = Image.open(io.BytesIO(await file.read()))
        image.load()
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the uploaded file as an image.")

    local = ImageSanitizer(redact_faces=redact_faces, redact_text=redact_text)
    result = local.sanitize(image, deep=deep, prompt_sanitizer=sanitizer if deep else None)

    report = {
        "faces_redacted": result.faces_redacted,
        "text_regions_redacted": result.text_regions_redacted,
        "pii_kinds_found": result.pii_kinds_found,
        "metadata_removed": result.metadata_removed,
        "warnings": result.warnings,
    }
    return Response(
        content=result.to_bytes("PNG"),
        media_type="image/png",
        headers={"X-Sanitization-Report": json.dumps(report)},
    )
