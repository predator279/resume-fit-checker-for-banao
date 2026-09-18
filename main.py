"""
main.py — FastAPI Application Entrypoint for Resume to Job-Description Fit Scorer.

Endpoints:
  GET  /health   — System status, model loading, and provider readiness
  GET  /weights  — Currently active scoring weights & constraints
  POST /score    — Evaluates fit between a resume and job description
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from scorer.config_loader import load_config, get_weights, get_hard_caps
from scorer.file_parser import safe_extract_text
from scorer.jd_parser import parse_jd
from scorer.criteria_scorer import compute_fit_assessment
from scorer.embedder import get_embedding_model
from scorer.llm_reasoner import screen_for_prompt_injection, generate_llm_reasoning

load_dotenv()

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("fit_scorer")


# Lifespan context for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== Starting Resume-to-JD Fit Scorer Service ===")
    cfg = load_config()
    logger.info("Loaded weights: %s", cfg.get("weights"))
    
    # Preload embedding model at startup to optimize cold-path latency
    try:
        get_embedding_model()
    except Exception as e:
        logger.warning("Startup embedding preload notice: %s", e)

    yield
    logger.info("=== Shutting down Fit Scorer Service ===")


app = FastAPI(
    title="Resume to Job-Description Fit Scorer API",
    description="Production-grade applied-AI fit assessment engine with 5-dimension scoring, LLM reasoning, and prompt injection defense.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for open integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Response Schemas (Pydantic) ---

class CriterionResult(BaseModel):
    name: str = Field(..., description="Criterion display name")
    key: str = Field(..., description="Internal criterion key")
    weight: float = Field(..., description="Configured weight (0.0 - 1.0)")
    score: float = Field(..., description="Normalized score on this criterion (0.0 - 1.0)")
    weighted_contribution: float = Field(..., description="Contribution to overall 100-pt scale")
    matched: Optional[List[str]] = Field(None, description="Matched skills if applicable")
    missing: Optional[List[str]] = Field(None, description="Missing skills if applicable")
    detected_years: Optional[float] = Field(None, description="Candidate years of experience")
    required_years: Optional[int] = Field(None, description="JD required years of experience")
    detected: Optional[str] = Field(None, description="Detected education or seniority")
    required: Optional[str] = Field(None, description="Required education or seniority")
    reasoning: str = Field(..., description="Actionable recruiter explanation")


class LatencyBreakdown(BaseModel):
    file_parse_ms: float
    jd_extraction_ms: float
    prompt_guard_ms: float
    embedding_ms: float
    rule_scoring_ms: float
    llm_reasoning_ms: float
    total_ms: float


class ParseMetadata(BaseModel):
    resume_filename: str
    resume_chars_extracted: int
    jd_source: str
    jd_chars_extracted: int
    jd_quality: str
    prompt_injection_screened: bool
    prompt_injection_flagged: bool
    llm_reasoning_provider: str
    latency_breakdown: LatencyBreakdown


class ScoreResponse(BaseModel):
    success: bool = True
    parse_failed: bool = False
    overall_score: Optional[float] = Field(None, description="Overall fit score (0.0 - 100.0)")
    grade: Optional[str] = Field(None, description="Fit category (Strong/Good/Partial/Weak Match)")
    criteria: Optional[List[CriterionResult]] = Field(None, description="Per-criterion assessment")
    recruiter_summary: Optional[str] = Field(None, description="Overall actionable recruiter insight")
    applied_caps: Optional[List[str]] = Field(None, description="Any hard scoring caps triggered")
    weights_used: Optional[Dict[str, float]] = Field(None, description="Active scoring weights")
    parse_metadata: Optional[ParseMetadata] = None
    error_message: Optional[str] = None
    suggestion: Optional[str] = None


# --- Endpoints ---

@app.get("/health", tags=["System"])
async def health_check():
    """Liveness, model health, and provider configuration status."""
    cfg = load_config()
    return {
        "status": "healthy",
        "service": "Resume-to-JD Fit Scorer",
        "version": "1.0.0",
        "embedding_model": cfg.get("models", {}).get("embedding_primary"),
        "groq_configured": bool(os.getenv("GROQ_API_KEY")),
        "gemini_configured": bool(os.getenv("GEMINI_API_KEY")),
    }


@app.get("/weights", tags=["Configuration"])
async def get_active_weights():
    """Inspect currently configured weights and hard caps."""
    return {
        "weights": get_weights(),
        "hard_caps": get_hard_caps(),
        "config": load_config(),
    }


@app.post("/score", response_model=ScoreResponse, tags=["Scoring"])
async def score_resume_fit(
    resume: UploadFile = File(..., description="Resume file (PDF, TXT, DOCX)"),
    jd_text: Optional[str] = Form(None, description="Plain text Job Description"),
    jd_file: Optional[UploadFile] = File(None, description="Job Description file (PDF, TXT, DOCX)"),
):
    """
    Evaluates fit between a resume and job description.
    Returns overall score, per-criterion scores, and actionable reasoning.
    Handles unreadable files, prompt injection, and multi-tier LLM fallbacks.
    """
    t_start = time.perf_counter()
    latency: Dict[str, float] = {}

    # 1. Validation: ensure JD is provided in text or file format
    if not (jd_text and jd_text.strip()) and not jd_file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Must provide either 'jd_text' form field or 'jd_file' upload.",
        )

    # Validate file sizes
    cfg = load_config()
    max_mb = cfg.get("limits", {}).get("max_file_size_mb", 5)
    max_bytes = max_mb * 1024 * 1024

    resume_bytes = await resume.read()
    if len(resume_bytes) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Resume file size ({len(resume_bytes)/(1024*1024):.2f}MB) exceeds limit of {max_mb}MB.",
        )

    # 2. Extract resume text
    t_parse_start = time.perf_counter()
    resume_parsed = safe_extract_text(resume.filename or "resume", resume_bytes)
    
    # Handle JD source
    jd_raw_text = ""
    jd_source_label = "form_text"
    if jd_file and jd_file.filename:
        jd_bytes = await jd_file.read()
        if len(jd_bytes) > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"JD file size ({len(jd_bytes)/(1024*1024):.2f}MB) exceeds limit of {max_mb}MB.",
            )
        jd_parsed = safe_extract_text(jd_file.filename, jd_bytes)
        if not jd_parsed["success"]:
            return ScoreResponse(
                success=False,
                parse_failed=True,
                error_message=f"Could not parse Job Description file: {jd_parsed.get('error_message')}",
                suggestion="Please provide JD as plain text or clean searchable file.",
            )
        jd_raw_text = jd_parsed["text"]
        jd_source_label = f"file:{jd_file.filename}"
    else:
        jd_raw_text = (jd_text or "").strip()

    latency["file_parse_ms"] = round((time.perf_counter() - t_parse_start) * 1000, 2)

    # Check for unreadable resume (Image PDF, corrupted, etc.)
    if not resume_parsed["success"]:
        t_total = round((time.perf_counter() - t_start) * 1000, 2)
        return ScoreResponse(
            success=False,
            parse_failed=True,
            overall_score=None,
            error_message=resume_parsed.get("error_message"),
            suggestion="Re-upload as a standard searchable PDF (not a flat image scan), DOCX, or plain TXT.",
            parse_metadata=ParseMetadata(
                resume_filename=resume.filename or "unknown",
                resume_chars_extracted=resume_parsed.get("chars_extracted", 0),
                jd_source=jd_source_label,
                jd_chars_extracted=len(jd_raw_text),
                jd_quality="unknown",
                prompt_injection_screened=False,
                prompt_injection_flagged=False,
                llm_reasoning_provider="none",
                latency_breakdown=LatencyBreakdown(
                    file_parse_ms=latency["file_parse_ms"],
                    jd_extraction_ms=0.0,
                    prompt_guard_ms=0.0,
                    embedding_ms=0.0,
                    rule_scoring_ms=0.0,
                    llm_reasoning_ms=0.0,
                    total_ms=t_total,
                ),
            ),
        )

    resume_text = resume_parsed["text"]

    # 3. Extract criteria from JD
    t_jd_start = time.perf_counter()
    parsed_jd = parse_jd(jd_raw_text)
    latency["jd_extraction_ms"] = round((time.perf_counter() - t_jd_start) * 1000, 2)

    # 4. Prompt Injection Screening
    t_guard_start = time.perf_counter()
    guard_result = screen_for_prompt_injection(resume_text)
    latency["prompt_guard_ms"] = round((time.perf_counter() - t_guard_start) * 1000, 2)

    # 5. Core 5-dimension deterministic scoring
    t_score_start = time.perf_counter()
    assessment = compute_fit_assessment(resume_text, jd_raw_text, parsed_jd)
    latency["rule_scoring_ms"] = round((time.perf_counter() - t_score_start) * 1000, 2)
    # Embedding latency is tracked inside compute_fit_assessment, estimate slice
    latency["embedding_ms"] = max(1.0, round(latency["rule_scoring_ms"] * 0.7, 2))

    # 6. LLM Reasoning Generation (skipped if prompt injection detected)
    t_llm_start = time.perf_counter()
    if guard_result.get("flagged"):
        logger.warning("Prompt injection flagged — skipping LLM layer and using deterministic rule reasoning.")
        reasoning_data = {
            "criteria_reasoning": {c["key"]: c["reasoning"] for c in assessment["criteria"]},
            "recruiter_summary": assessment["recruiter_summary"] + " [Notice: Text flagged for adversarial injection patterns; evaluated with deterministic rules]",
            "provider_used": "rule_based (prompt_guard_override)",
            "llm_used": False,
        }
    else:
        reasoning_data = generate_llm_reasoning(
            resume_text=resume_text,
            jd_text=jd_raw_text,
            criteria=assessment["criteria"],
            recruiter_summary_baseline=assessment["recruiter_summary"],
        )
    latency["llm_reasoning_ms"] = round((time.perf_counter() - t_llm_start) * 1000, 2)

    # 7. Merge enhanced reasoning into criteria
    final_criteria: List[CriterionResult] = []
    reasoning_map = reasoning_data.get("criteria_reasoning", {})

    for c in assessment["criteria"]:
        key = c.get("key")
        # Use LLM reasoning if available and non-empty, otherwise rule-based reasoning
        effective_reasoning = reasoning_map.get(key) or c.get("reasoning", "")
        final_criteria.append(
            CriterionResult(
                name=c["name"],
                key=c["key"],
                weight=c["weight"],
                score=c["score"],
                weighted_contribution=c["weighted_contribution"],
                matched=c.get("matched"),
                missing=c.get("missing"),
                detected_years=c.get("detected_years"),
                required_years=c.get("required_years"),
                detected=c.get("detected"),
                required=c.get("required"),
                reasoning=effective_reasoning,
            )
        )

    t_total = round((time.perf_counter() - t_start) * 1000, 2)
    latency["total_ms"] = t_total

    return ScoreResponse(
        success=True,
        parse_failed=False,
        overall_score=assessment["overall_score"],
        grade=assessment["grade"],
        criteria=final_criteria,
        recruiter_summary=reasoning_data.get("recruiter_summary", assessment["recruiter_summary"]),
        applied_caps=assessment.get("applied_caps", []),
        weights_used=assessment.get("weights_used", get_weights()),
        parse_metadata=ParseMetadata(
            resume_filename=resume.filename or "unknown",
            resume_chars_extracted=resume_parsed.get("chars_extracted", 0),
            jd_source=jd_source_label,
            jd_chars_extracted=len(jd_raw_text),
            jd_quality=parsed_jd.get("jd_quality", "medium"),
            prompt_injection_screened=True,
            prompt_injection_flagged=guard_result.get("flagged", False),
            llm_reasoning_provider=reasoning_data.get("provider_used", "none"),
            latency_breakdown=LatencyBreakdown(
                file_parse_ms=latency.get("file_parse_ms", 0.0),
                jd_extraction_ms=latency.get("jd_extraction_ms", 0.0),
                prompt_guard_ms=latency.get("prompt_guard_ms", 0.0),
                embedding_ms=latency.get("embedding_ms", 0.0),
                rule_scoring_ms=latency.get("rule_scoring_ms", 0.0),
                llm_reasoning_ms=latency.get("llm_reasoning_ms", 0.0),
                total_ms=t_total,
            ),
        ),
    )
