"""
llm_reasoner.py — Resilient LLM Reasoning & Prompt Guard Layer.

Features:
  1. Prompt injection screening via Llama Prompt Guard 2 & heuristic regex.
  2. Multi-tier LLM Provider Fallback:
       Tier 1: Groq Cloud (openai/gpt-oss-20b)
       Tier 2: Groq Cloud (openai/gpt-oss-120b)
       Tier 3: Google Gemini (gemini-2.0-flash-lite)
       Tier 4: Rule-based deterministic fallback strings (zero API dependence)
  3. Robust JSON extraction & schema validation.
  4. Non-blocking error handling: LLM exceptions never fail the scoring response.
"""

import json
import logging
import os
import re
import time
from typing import Dict, Any, List, Optional
import httpx

from .config_loader import load_config

logger = logging.getLogger(__name__)

# Heuristic prompt injection patterns
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior)\s+instructions?",
    r"disregard\s+(all\s+)?(previous|prior)\s+instructions?",
    r"system\s*:\s*override",
    r"you\s+are\s+now\s+a",
    r"give\s+this\s+(candidate|resume)\s+(a\s+)?(score\s+of\s+)?(100|perfect)",
    r"override\s+scoring",
    r"ignore\s+the\s+job\s+description",
    r"bypass\s+ats",
]


def screen_for_prompt_injection(text: str) -> Dict[str, Any]:
    """
    Screens input text for prompt injection attempts using both heuristic regex
    and Llama Prompt Guard 2 via Groq (if available).
    """
    if not text:
        return {"flagged": False, "method": "none"}

    # 1. Fast heuristic regex check
    text_lower = text.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text_lower):
            logger.warning("Prompt injection detected via heuristic filter: '%s'", pattern)
            return {
                "flagged": True,
                "method": "heuristic_regex",
                "matched_pattern": pattern,
            }

    # 2. Llama Prompt Guard screening via Groq (if API key provided)
    groq_api_key = os.getenv("GROQ_API_KEY")
    if groq_api_key:
        try:
            cfg = load_config()
            guard_model = cfg.get("models", {}).get("groq_guard_model", "meta-llama/llama-prompt-guard-2-86m")
            headers = {
                "Authorization": f"Bearer {groq_api_key}",
                "Content-Type": "application/json",
            }
            # Truncate text snippet for guard screening
            snippet = text[:2000]
            payload = {
                "model": guard_model,
                "messages": [
                    {"role": "user", "content": snippet}
                ],
                "temperature": 0.0,
                "max_tokens": 10,
            }
            with httpx.Client(timeout=4.0) as client:
                resp = client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=payload,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    guard_output = data["choices"][0]["message"]["content"].strip().lower()
                    if "jailbreak" in guard_output or "injection" in guard_output:
                        logger.warning("Prompt injection flagged by %s: %s", guard_model, guard_output)
                        return {
                            "flagged": True,
                            "method": guard_model,
                            "output": guard_output,
                        }
        except Exception as e:
            logger.debug("Prompt guard API call skipped or timed out: %s", e)

    return {"flagged": False, "method": "clean"}


def generate_llm_reasoning(
    resume_text: str,
    jd_text: str,
    criteria: List[Dict[str, Any]],
    recruiter_summary_baseline: str,
) -> Dict[str, Any]:
    """
    Generates tailored, professional natural-language recruiter reasoning for each criterion.
    Uses multi-tier fallback (Groq -> Gemini -> Rule-based).

    Returns:
        {
            "criteria_reasoning": { "skills_match": str, ... },
            "recruiter_summary": str,
            "provider_used": str,
            "llm_used": bool
        }
    """
    cfg = load_config()
    max_r = cfg.get("limits", {}).get("max_resume_chars_llm", 3500)
    max_jd = cfg.get("limits", {}).get("max_jd_chars_llm", 2000)

    # Sanitize & truncate context
    resume_snippet = resume_text[:max_r]
    jd_snippet = jd_text[:max_jd]

    # Prepare structured summary of scores for LLM context
    criteria_summary = []
    for c in criteria:
        criteria_summary.append({
            "key": c.get("key"),
            "name": c.get("name"),
            "score": c.get("score"),
            "weight": c.get("weight"),
            "baseline_note": c.get("reasoning"),
            "matched_skills": c.get("matched", []),
            "missing_skills": c.get("missing", []),
            "detected_years": c.get("detected_years"),
            "required_years": c.get("required_years"),
        })

    prompt_payload = {
        "job_description_snippet": jd_snippet,
        "resume_snippet": resume_snippet,
        "evaluated_criteria": criteria_summary,
    }

    system_prompt = (
        "You are an expert AI recruiting assistant. Analyze the candidate's criteria evaluation against the job description.\n"
        "Generate concise, professional recruiter reasoning for EACH of the 5 criteria, plus an overall recruiter summary.\n"
        "STRICT REQUIREMENTS:\n"
        "1. Do not alter or hallucinate new criteria. Keep scores unchanged.\n"
        "2. Provide exactly 1-2 factual, actionable sentences per criterion explaining WHY that score was achieved.\n"
        "3. Output MUST be valid, clean JSON with this exact schema:\n"
        "{\n"
        '  "skills_match": "Reasoning for skills...",\n'
        '  "semantic_similarity": "Reasoning for semantic alignment...",\n'
        '  "experience_match": "Reasoning for experience...",\n'
        '  "education_match": "Reasoning for education...",\n'
        '  "seniority_fit": "Reasoning for seniority...",\n'
        '  "recruiter_summary": "Overall 1-2 sentence recruiter assessment."\n'
        "}"
    )

    user_prompt = (
        f"Criteria & Context:\n{json.dumps(prompt_payload, indent=2)}\n\n"
        "Generate the JSON assessment object now."
    )

    # Tier 1 & 2: Groq Cloud (gpt-oss-20b -> gpt-oss-120b)
    groq_api_key = os.getenv("GROQ_API_KEY")
    if groq_api_key:
        for model_key in ("groq_primary_llm", "groq_secondary_llm"):
            model_name = cfg.get("models", {}).get(model_key, "openai/gpt-oss-20b")
            result = _call_groq(groq_api_key, model_name, system_prompt, user_prompt, cfg)
            if result:
                return {
                    "criteria_reasoning": result.get("reasoning_map", {}),
                    "recruiter_summary": result.get("recruiter_summary", recruiter_summary_baseline),
                    "provider_used": f"groq:{model_name}",
                    "llm_used": True,
                }

    # Tier 3: Google Gemini (gemini-2.0-flash-lite)
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if gemini_api_key:
        gemini_model = cfg.get("models", {}).get("gemini_fallback_model", "gemini-2.0-flash-lite")
        result = _call_gemini(gemini_api_key, gemini_model, system_prompt, user_prompt, cfg)
        if result:
            return {
                "criteria_reasoning": result.get("reasoning_map", {}),
                "recruiter_summary": result.get("recruiter_summary", recruiter_summary_baseline),
                "provider_used": f"gemini:{gemini_model}",
                "llm_used": True,
            }

    # Tier 4: Rule-based fallback
    logger.info("Using rule-based reasoning fallback (no active LLM provider succeeded).")
    rule_map = {c.get("key", ""): c.get("reasoning", "") for c in criteria}
    return {
        "criteria_reasoning": rule_map,
        "recruiter_summary": recruiter_summary_baseline,
        "provider_used": "rule_based_deterministic",
        "llm_used": False,
    }


def _call_groq(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    cfg: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Call Groq chat completion API with retry."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    llm_cfg = cfg.get("llm", {})
    timeout = llm_cfg.get("timeout_seconds", 12)
    max_retries = llm_cfg.get("retry_attempts", 1)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": llm_cfg.get("temperature", 0.0),
        "max_tokens": llm_cfg.get("max_tokens", 600),
        "response_format": {"type": "json_object"},
    }

    for attempt in range(max_retries + 1):
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=payload,
                )
                if resp.status_code == 200:
                    raw_content = resp.json()["choices"][0]["message"]["content"]
                    parsed = _parse_json_safely(raw_content)
                    if parsed:
                        return parsed
                elif resp.status_code == 429:
                    logger.warning("Groq rate limited on %s (attempt %s).", model, attempt + 1)
                    time.sleep(1.5 * (attempt + 1))
                else:
                    logger.warning("Groq API error (%s): %s", resp.status_code, resp.text[:200])
        except Exception as e:
            logger.warning("Groq call exception on %s (attempt %s): %s", model, attempt + 1, e)
            time.sleep(1.0)

    return None


def _call_gemini(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    cfg: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Call Google Gemini REST API."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": f"{system_prompt}\n\n{user_prompt}"}]}
        ],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 600,
            "responseMimeType": "application/json",
        },
    }

    try:
        with httpx.Client(timeout=14.0) as client:
            resp = client.post(url, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
                return _parse_json_safely(raw_text)
            else:
                logger.warning("Gemini API error (%s): %s", resp.status_code, resp.text[:200])
    except Exception as e:
        logger.warning("Gemini API exception: %s", e)

    return None


def _parse_json_safely(raw_text: str) -> Optional[Dict[str, Any]]:
    """
    Robust JSON extraction supporting Markdown code fences and raw objects.
    """
    if not raw_text:
        return None

    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            summary = data.pop("recruiter_summary", None)
            return {
                "reasoning_map": data,
                "recruiter_summary": summary,
            }
    except Exception:
        pass

    # Regex extraction fallback if JSON has trailing commas or text
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict):
                summary = data.pop("recruiter_summary", None)
                return {
                    "reasoning_map": data,
                    "recruiter_summary": summary,
                }
        except Exception:
            pass

    return None
