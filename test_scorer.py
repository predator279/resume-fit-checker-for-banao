"""
test_scorer.py — Automated test and calibration verification suite.
Runs 3 calibration samples, unreadable file edge case, and adversarial injection test.
Measures latency breakdown and validates scoring calibration consistency.
"""

import os
import json
from pathlib import Path
from scorer.config_loader import load_config
from scorer.file_parser import safe_extract_text
from scorer.jd_parser import parse_jd
from scorer.criteria_scorer import compute_fit_assessment
from scorer.llm_reasoner import screen_for_prompt_injection, generate_llm_reasoning

BASE_DIR = Path(__file__).parent
SAMPLES_DIR = BASE_DIR / "samples"


def run_test_suite():
    print("================================================================")
    print("  RESUME-TO-JD FIT SCORER — TEST & CALIBRATION BENCHMARK SUITE  ")
    print("================================================================\n")

    cfg = load_config()
    print(f"Loaded Weights: {cfg['weights']}\n")

    jd_path = SAMPLES_DIR / "jd_senior_backend.txt"
    with open(jd_path, "r", encoding="utf-8") as f:
        jd_text = f.read()

    print(f"=== Loaded Job Description: {jd_path.name} ===")
    parsed_jd = parse_jd(jd_text)
    print(f"  Extracted Skills: {parsed_jd['skills']}")
    print(f"  Required Experience: {parsed_jd['experience_years']} years")
    print(f"  Required Education: {parsed_jd['education']}")
    print(f"  Required Seniority: {parsed_jd['seniority']}")
    print(f"  JD Quality Score: {parsed_jd['jd_quality']}\n")

    test_files = [
        ("Sample 1 (Strong Senior Match)", "resume_sample_1_strong.txt"),
        ("Sample 2 (Moderate Mid Match)", "resume_sample_2_similar_moderate.txt"),
        ("Sample 3 (Weak Frontend Match)", "resume_sample_3_weak.txt"),
        ("Adversarial Injection Test", "resume_adversarial_injection.txt"),
    ]

    scores = {}

    for label, filename in test_files:
        filepath = SAMPLES_DIR / filename
        with open(filepath, "rb") as f:
            raw_bytes = f.read()

        print(f"----------------------------------------------------------------")
        print(f"Testing: {label} ({filename})")
        print(f"----------------------------------------------------------------")

        # 1. Parse file
        parsed = safe_extract_text(filename, raw_bytes)
        if not parsed["success"]:
            print(f"  [!] Parse Failed: {parsed['error_message']}")
            continue

        resume_text = parsed["text"]

        # 2. Prompt guard check
        guard_res = screen_for_prompt_injection(resume_text)
        print(f"  Prompt Guard Check: Flagged = {guard_res['flagged']} (Method: {guard_res.get('method')})")

        # 3. Compute fit assessment
        assessment = compute_fit_assessment(resume_text, jd_text, parsed_jd)
        scores[filename] = assessment["overall_score"]

        print(f"  Overall Fit Score: {assessment['overall_score']} / 100.0 ({assessment['grade']})")
        print(f"  Applied Caps: {assessment['applied_caps'] or 'None'}")
        print("  Criteria Breakdown:")
        for c in assessment["criteria"]:
            print(f"    - {c['name']} (Weight: {c['weight']}): Score = {c['score']} -> Contrib = {c['weighted_contribution']} pts")
            print(f"      Reasoning: {c['reasoning']}")

        print(f"  Recruiter Summary: {assessment['recruiter_summary']}\n")

    # 4. Calibration Check
    s1 = scores.get("resume_sample_1_strong.txt", 0)
    s2 = scores.get("resume_sample_2_similar_moderate.txt", 0)
    s3 = scores.get("resume_sample_3_weak.txt", 0)
    diff_similar = abs(s1 - s2)

    print("================================================================")
    print("  CALIBRATION VERIFICATION & CONSISTENCY ANALYSIS               ")
    print("================================================================")
    print(f"  Sample 1 (Strong Senior):    {s1} pts")
    print(f"  Sample 2 (Moderate Mid):     {s2} pts")
    print(f"  Sample 3 (Weak Frontend):    {s3} pts")
    print(f"  Score Gap Between S1 and S2: {diff_similar:.1f} pts")

    if diff_similar < 40.0:
        print(f"  [PASS] Calibration constraint satisfied! (Gap {diff_similar:.1f} pts is well below the 40-pt limit).")
    else:
        print(f"  [FAIL] Calibration gap too large: {diff_similar:.1f} pts >= 40.0 pts.")

    # 5. Unreadable File Test
    print("\n================================================================")
    print("  EDGE CASE TEST: UNREADABLE / EMPTY RESUME                     ")
    print("================================================================")
    empty_parse = safe_extract_text("corrupted_scan.pdf", b"")
    print(f"  Empty file parse success: {empty_parse['success']}")
    print(f"  Error message: {empty_parse['error_message']}")

    tiny_parse = safe_extract_text("scanned_image.pdf", b"%PDF-1.4 image only dummy content")
    print(f"  Scanned image parse success: {tiny_parse['success']}")
    print(f"  Error message: {tiny_parse['error_message']}")
    print("================================================================\n")


if __name__ == "__main__":
    run_test_suite()
