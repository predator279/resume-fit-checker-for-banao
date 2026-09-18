"""
criteria_scorer.py — 5-dimension deterministic ATS criteria evaluation.
Evaluates:
  1. Skills / Keywords Match (frequency weighted)
  2. Semantic Similarity (MiniLM / BGE embeddings)
  3. Experience Match (years detected vs required)
  4. Education Match (degree level hierarchy + STEM relevance)
  5. Seniority Fit (title-level match vs JD target)

Weights and hard-caps are dynamically loaded from config/weights.yaml.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple

from .config_loader import get_weights, get_hard_caps
from .embedder import calculate_semantic_similarity
from .jd_parser import DEGREE_LEVELS, DEGREE_LABEL, SENIORITY_ORDER

logger = logging.getLogger(__name__)

RESUME_SENIORITY_SIGNALS = {
    "intern": "intern",
    "trainee": "intern",
    "fresher": "junior",
    "junior": "junior",
    "associate": "junior",
    "entry level": "junior",
    "entry-level": "junior",
    "mid level": "mid",
    "mid-level": "mid",
    "ii": "mid",
    "iii": "senior",
    "senior": "senior",
    "sr.": "senior",
    "sr ": "senior",
    "staff": "senior",
    "lead": "lead",
    "principal": "principal",
    "architect": "principal",
    "manager": "lead",
    "director": "principal",
    "head of": "principal",
    "vp": "principal",
    "vice president": "principal",
}

STEM_KEYWORDS = [
    "computer science", "information technology", "software", "engineering",
    "electronics", "electrical", "data science", "artificial intelligence",
    "machine learning", "mathematics", "statistics", "cs", "it", "cse", "ece",
]


def score_keywords(resume_text: str, required_skills: List[str]) -> Dict[str, Any]:
    """
    Frequency-weighted keyword matching.
    0 occurrences  -> 0.0
    1 occurrence   -> 0.7  (mentioned)
    2+ occurrences -> 1.0  (established / practiced)
    """
    if not required_skills:
        return {
            "score": 0.5,
            "matched": [],
            "missing": [],
            "reasoning": "No specific technical skills specified in JD; evaluated neutrally.",
        }

    resume_lower = resume_text.lower()
    matched = []
    missing = []
    total = 0.0

    for skill in required_skills:
        pattern = r"(?<![a-zA-Z0-9+#.])" + re.escape(skill.lower()) + r"(?![a-zA-Z0-9+#.])"
        count = len(re.findall(pattern, resume_lower))
        if count >= 2:
            total += 1.0
            matched.append(skill)
        elif count == 1:
            total += 0.7
            matched.append(skill)
        else:
            total += 0.0
            missing.append(skill)

    score = round(total / len(required_skills), 4)

    if score >= 0.8:
        reasoning = f"Strong skills alignment: matched {len(matched)}/{len(required_skills)} required skills ({', '.join(matched[:4])})."
    elif score >= 0.5:
        reasoning = f"Moderate skills match: found {len(matched)}/{len(required_skills)} skills. Missing: {', '.join(missing[:3])}."
    else:
        reasoning = f"Significant skill gaps: missing critical requirements ({', '.join(missing[:4])})."

    return {
        "score": score,
        "matched": matched,
        "missing": missing,
        "reasoning": reasoning,
    }


def score_semantic(resume_text: str, jd_text: str) -> Dict[str, Any]:
    """
    Evaluates semantic closeness between candidate profile and JD context.
    """
    sim_res = calculate_semantic_similarity(resume_text, jd_text)
    score = sim_res["score"]

    if score >= 0.75:
        reasoning = "High contextual alignment with the overall job scope and domain responsibilities."
    elif score >= 0.55:
        reasoning = "Moderate semantic similarity; general domain matches though specific context diverges."
    else:
        reasoning = "Low semantic alignment with the targeted job role and responsibilities."

    return {
        "score": score,
        "model_used": sim_res.get("model_used"),
        "fallback_used": sim_res.get("fallback_used", False),
        "reasoning": reasoning,
    }


def extract_years_of_experience(text: str) -> Optional[float]:
    """Extract maximum verified experience in years from date spans and explicit statements."""
    t = text.lower()
    candidates = []

    # Date ranges: "2019 - 2024" or "2021 - Present"
    for m in re.finditer(r"(\d{4})\s*[-–—]\s*((?:\d{4}|present|current|now|till date))", t):
        start_yr = int(m.group(1))
        end_raw = m.group(2).strip()
        if end_raw.isdigit():
            end_yr = int(end_raw)
        else:
            end_yr = 2025
        diff = end_yr - start_yr
        if 0 < diff < 45:
            candidates.append(float(diff))

    # Explicit phrases: "5+ years", "3 years of experience"
    for pat in [
        r"(\d+(?:\.\d+)?)\s*\+?\s*years?",
        r"(\d+(?:\.\d+)?)\s*\+?\s*yrs?",
    ]:
        for m in re.finditer(pat, t):
            val = float(m.group(1))
            if 0 < val < 45:
                candidates.append(val)

    # Range mentions: "3-5 years"
    for m in re.finditer(r"(\d+)\s*-\s*(\d+)\s*years?", t):
        candidates.append(float(m.group(2)))

    if not candidates:
        return None
    # Cap upper bound at 25 to prevent exaggerated claims from skewing
    return min(25.0, max(candidates))


def score_experience(resume_text: str, required_years: Optional[int]) -> Dict[str, Any]:
    """Score candidate experience against required years."""
    detected = extract_years_of_experience(resume_text)

    if required_years is None:
        return {
            "score": 0.5,
            "detected_years": detected,
            "required_years": None,
            "reasoning": "No minimum years of experience specified in JD; neutral score assigned.",
        }

    if detected is None:
        score = 0.3
        reasoning = f"Could not detect clear years of experience in resume; required {required_years}+ years."
    elif detected >= required_years:
        score = 1.0
        reasoning = f"Meets/exceeds experience requirement ({round(detected, 1)} years detected vs {required_years} years required)."
    elif detected >= required_years - 1:
        score = 0.7
        reasoning = f"Borderline experience ({round(detected, 1)} years detected vs {required_years} years required)."
    elif detected >= required_years - 2:
        score = 0.4
        reasoning = f"Under-experienced for this role ({round(detected, 1)} years detected vs {required_years} years required)."
    else:
        score = 0.2
        reasoning = f"Significant experience deficit ({round(detected, 1)} years detected vs {required_years} years required)."

    return {
        "score": round(score, 4),
        "detected_years": round(detected, 1) if detected else None,
        "required_years": required_years,
        "reasoning": reasoning,
    }


def detect_degree_level(text: str) -> Tuple[int, str]:
    t = text.lower()
    best_level, best_label = 0, "Not detected"
    for kw, level in DEGREE_LEVELS.items():
        if re.search(r"(?<![a-z])" + re.escape(kw) + r"(?![a-z])", t) and level > best_level:
            best_level = level
            best_label = DEGREE_LABEL.get(level, kw)
    return best_level, best_label


def score_education(resume_text: str, required_education: Optional[str]) -> Dict[str, Any]:
    """Score education degree hierarchy and STEM relevance."""
    detected_level, detected_label = detect_degree_level(resume_text)

    if not required_education:
        return {
            "score": 0.5,
            "detected": detected_label,
            "required": "Not specified",
            "reasoning": "No specific education requirement specified in JD.",
        }

    req_level, req_label = detect_degree_level(required_education)

    if req_level == 0:
        return {
            "score": 0.5,
            "detected": detected_label,
            "required": required_education,
            "reasoning": f"Education requirement parsed as {required_education}.",
        }

    if detected_level == 0:
        score = 0.2
        reasoning = f"No formal degree detected; JD specifies {req_label}."
    elif detected_level >= req_level:
        score = 1.0
        reasoning = f"Education requirement met: {detected_label} (required {req_label})."
    elif detected_level == req_level - 1:
        score = 0.6
        reasoning = f"Slightly below target degree level ({detected_label} vs required {req_label})."
    else:
        score = 0.2
        reasoning = f"Education level below requirement ({detected_label} vs required {req_label})."

    # STEM alignment bonus
    resume_lower = resume_text.lower()
    if any(kw in resume_lower for kw in STEM_KEYWORDS):
        score = min(1.0, score + 0.1)

    return {
        "score": round(score, 4),
        "detected": detected_label,
        "required": req_label,
        "reasoning": reasoning,
    }


def detect_resume_seniority(resume_text: str) -> Optional[str]:
    t = resume_text.lower()
    detected = []
    for kw, level in RESUME_SENIORITY_SIGNALS.items():
        if kw in t:
            detected.append(level)
    if not detected:
        return None
    for lvl in reversed(SENIORITY_ORDER):
        if lvl in detected:
            return lvl
    return detected[-1]


def score_seniority(resume_text: str, required_seniority: Optional[str]) -> Dict[str, Any]:
    """Score candidate seniority positioning against JD target."""
    detected_raw = detect_resume_seniority(resume_text)
    detected = detected_raw.lower() if detected_raw else None
    required = required_seniority.lower() if required_seniority else None

    if not required:
        return {
            "score": 0.7,
            "detected": detected_raw.capitalize() if detected_raw else "Not detected",
            "required": "Not specified",
            "mismatch": False,
            "reasoning": "No specific seniority tier specified in JD; neutral-positive baseline.",
        }

    if not detected:
        score = 0.4
        mismatch = False
        reasoning = "Candidate title history does not clearly indicate seniority level."
    else:
        req_idx = SENIORITY_ORDER.index(required) if required in SENIORITY_ORDER else 2
        det_idx = SENIORITY_ORDER.index(detected) if detected in SENIORITY_ORDER else 2
        gap = det_idx - req_idx

        if gap == 0:
            score = 1.0
            reasoning = f"Exact seniority match ({detected.capitalize()} level)."
        elif gap == 1:
            score = 0.85
            reasoning = f"Seniority slightly above target ({detected.capitalize()} vs {required.capitalize()})."
        elif gap == -1:
            score = 0.55
            reasoning = f"Seniority one step below target ({detected.capitalize()} vs {required.capitalize()})."
        elif gap >= 2:
            score = 0.80
            reasoning = f"Candidate is overqualified ({detected.capitalize()} vs {required.capitalize()})."
        else:
            score = 0.20
            reasoning = f"Candidate is significantly under-qualified for senior-tier role ({detected.capitalize()} vs {required.capitalize()})."

        mismatch = (
            required in ("senior", "lead", "principal")
            and detected in ("intern", "junior")
        )

    return {
        "score": round(score, 4),
        "detected": detected_raw.capitalize() if detected_raw else "Not detected",
        "required": required_seniority.capitalize() if required_seniority else "Not specified",
        "mismatch": mismatch,
        "reasoning": reasoning,
    }


def compute_fit_assessment(
    resume_text: str,
    jd_text: str,
    parsed_jd: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Computes overall fit score and per-criterion assessment.
    Applies configurable weights and hard caps.
    """
    weights = get_weights()
    hard_caps = get_hard_caps()

    # 1. Evaluate 5 criteria
    kw_res = score_keywords(resume_text, parsed_jd.get("skills", []))
    sem_res = score_semantic(resume_text, jd_text)
    exp_res = score_experience(resume_text, parsed_jd.get("experience_years"))
    edu_res = score_education(resume_text, parsed_jd.get("education"))
    sen_res = score_seniority(resume_text, parsed_jd.get("seniority"))

    # 2. Weighted score calculation
    w_kw = weights.get("skills_match", 0.35)
    w_sem = weights.get("semantic_similarity", 0.20)
    w_exp = weights.get("experience_match", 0.25)
    w_edu = weights.get("education_match", 0.10)
    w_sen = weights.get("seniority_fit", 0.10)

    contrib_kw = round(kw_res["score"] * w_kw * 100, 2)
    contrib_sem = round(sem_res["score"] * w_sem * 100, 2)
    contrib_exp = round(exp_res["score"] * w_exp * 100, 2)
    contrib_edu = round(edu_res["score"] * w_edu * 100, 2)
    contrib_sen = round(sen_res["score"] * w_sen * 100, 2)

    raw_overall = contrib_kw + contrib_sem + contrib_exp + contrib_edu + contrib_sen
    final_score = raw_overall
    applied_caps = []

    # 3. Hard caps evaluation
    low_skills_thresh = hard_caps.get("low_skills_threshold", 0.25)
    low_skills_cap = hard_caps.get("low_skills_max_score", 45.0)
    if kw_res["score"] < low_skills_thresh:
        final_score = min(final_score, low_skills_cap)
        applied_caps.append(f"Capped at {low_skills_cap} due to low skills match (<{int(low_skills_thresh*100)}%).")

    seniority_cap = hard_caps.get("seniority_mismatch_max_score", 60.0)
    if sen_res.get("mismatch"):
        final_score = min(final_score, seniority_cap)
        applied_caps.append(f"Capped at {seniority_cap} due to severe seniority mismatch.")

    # 4. Formulate structured criteria list
    criteria = [
        {
            "name": "Skills Match",
            "key": "skills_match",
            "weight": w_kw,
            "score": kw_res["score"],
            "weighted_contribution": contrib_kw,
            "matched": kw_res["matched"],
            "missing": kw_res["missing"],
            "reasoning": kw_res["reasoning"],
        },
        {
            "name": "Semantic Similarity",
            "key": "semantic_similarity",
            "weight": w_sem,
            "score": sem_res["score"],
            "weighted_contribution": contrib_sem,
            "model_used": sem_res.get("model_used"),
            "reasoning": sem_res["reasoning"],
        },
        {
            "name": "Experience Match",
            "key": "experience_match",
            "weight": w_exp,
            "score": exp_res["score"],
            "weighted_contribution": contrib_exp,
            "detected_years": exp_res["detected_years"],
            "required_years": exp_res["required_years"],
            "reasoning": exp_res["reasoning"],
        },
        {
            "name": "Education Match",
            "key": "education_match",
            "weight": w_edu,
            "score": edu_res["score"],
            "weighted_contribution": contrib_edu,
            "detected": edu_res["detected"],
            "required": edu_res["required"],
            "reasoning": edu_res["reasoning"],
        },
        {
            "name": "Seniority Fit",
            "key": "seniority_fit",
            "weight": w_sen,
            "score": sen_res["score"],
            "weighted_contribution": contrib_sen,
            "detected": sen_res["detected"],
            "required": sen_res["required"],
            "reasoning": sen_res["reasoning"],
        },
    ]

    # Recruiter summary recommendation
    if final_score >= 80:
        grade = "Strong Match"
        recruiter_summary = "Strong candidate match. Exceeds core technical and experience requirements; highly recommended for interview."
    elif final_score >= 65:
        grade = "Good Match"
        recruiter_summary = "Good candidate match. Meets primary requirements with minor gaps; recommended for screening."
    elif final_score >= 45:
        grade = "Partial Match"
        recruiter_summary = "Partial fit. Several notable gaps in skills, experience, or seniority. Review specific criteria before advancing."
    else:
        grade = "Weak Match"
        recruiter_summary = "Weak fit. Does not meet minimum qualifications or core technical competencies."

    if applied_caps:
        recruiter_summary += f" ({'; '.join(applied_caps)})"

    return {
        "overall_score": round(final_score, 1),
        "raw_overall_score": round(raw_overall, 1),
        "grade": grade,
        "criteria": criteria,
        "recruiter_summary": recruiter_summary,
        "applied_caps": applied_caps,
        "weights_used": weights,
    }
