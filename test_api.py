"""
test_api.py — API Integration and End-to-End Endpoint Verification.
Uses FastAPI TestClient to test /health, /weights, /score (with text & file JD, unreadable PDF, adversarial).
"""

from pathlib import Path
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)
SAMPLES_DIR = Path(__file__).parent / "samples"


def test_endpoints():
    print("================================================================")
    print("  FASTAPI ENDPOINT INTEGRATION & LATENCY VERIFICATION           ")
    print("================================================================\n")

    # 1. Health check
    print("1. Testing GET /health...")
    resp = client.get("/health")
    assert resp.status_code == 200, f"Health check failed: {resp.text}"
    health_data = resp.json()
    print(f"   [PASS] Health Status: {health_data}\n")

    # 2. Weights check
    print("2. Testing GET /weights...")
    resp = client.get("/weights")
    assert resp.status_code == 200, f"Weights check failed: {resp.text}"
    weights_data = resp.json()
    print(f"   [PASS] Active Weights: {weights_data['weights']}\n")

    # Load JD text
    with open(SAMPLES_DIR / "jd_senior_backend.txt", "r", encoding="utf-8") as f:
        jd_text = f.read()

    # 3. Score Endpoint with Strong Resume
    print("3. Testing POST /score with Strong Resume + Form JD...")
    with open(SAMPLES_DIR / "resume_sample_1_strong.txt", "rb") as f:
        resume_bytes = f.read()

    resp = client.post(
        "/score",
        files={"resume": ("alex_morgan_resume.txt", resume_bytes, "text/plain")},
        data={"jd_text": jd_text},
    )
    assert resp.status_code == 200, f"Score failed: {resp.text}"
    score_data = resp.json()
    assert score_data["success"] is True
    assert score_data["overall_score"] > 80.0
    print(f"   [PASS] Overall Score: {score_data['overall_score']} ({score_data['grade']})")
    print(f"   [PASS] Latency Breakdown (ms): {score_data['parse_metadata']['latency_breakdown']}")
    print(f"   [PASS] Provider Used: {score_data['parse_metadata']['llm_reasoning_provider']}\n")

    # 4. Score Endpoint with JD uploaded as a file
    print("4. Testing POST /score with JD uploaded as a file...")
    with open(SAMPLES_DIR / "jd_senior_backend.txt", "rb") as f:
        jd_bytes = f.read()

    resp = client.post(
        "/score",
        files={
            "resume": ("jordan_lee_resume.txt", resume_bytes, "text/plain"),
            "jd_file": ("job_description.txt", jd_bytes, "text/plain"),
        },
    )
    assert resp.status_code == 200, f"Score with file JD failed: {resp.text}"
    file_jd_data = resp.json()
    assert file_jd_data["success"] is True
    print(f"   [PASS] JD file source: {file_jd_data['parse_metadata']['jd_source']}")
    print(f"   [PASS] Overall Score: {file_jd_data['overall_score']}\n")

    # 5. Score Endpoint with Unreadable / Corrupted Resume
    print("5. Testing POST /score with Unreadable Resume (0 bytes)...")
    resp = client.post(
        "/score",
        files={"resume": ("corrupted_scan.pdf", b"", "application/pdf")},
        data={"jd_text": jd_text},
    )
    assert resp.status_code == 200
    unreadable_data = resp.json()
    assert unreadable_data["parse_failed"] is True
    assert unreadable_data["overall_score"] is None
    print(f"   [PASS] Handled unreadable file gracefully (HTTP 200 with parse_failed=True):")
    print(f"          Error: {unreadable_data['error_message']}")
    print(f"          Suggestion: {unreadable_data['suggestion']}\n")

    # 6. Score Endpoint with Adversarial Injection Resume
    print("6. Testing POST /score with Adversarial Prompt Injection Resume...")
    with open(SAMPLES_DIR / "resume_adversarial_injection.txt", "rb") as f:
        adv_bytes = f.read()

    resp = client.post(
        "/score",
        files={"resume": ("adversarial_resume.txt", adv_bytes, "text/plain")},
        data={"jd_text": jd_text},
    )
    assert resp.status_code == 200
    adv_data = resp.json()
    assert adv_data["parse_metadata"]["prompt_injection_flagged"] is True
    assert adv_data["overall_score"] < 45.0
    print(f"   [PASS] Prompt injection flagged: {adv_data['parse_metadata']['prompt_injection_flagged']}")
    print(f"   [PASS] Overall score kept safe at: {adv_data['overall_score']} (immune to override)\n")

    print("================================================================")
    print("  ALL FASTAPI INTEGRATION TESTS PASSED SUCCESSFULLY!            ")
    print("================================================================\n")


if __name__ == "__main__":
    test_endpoints()
