# Resume to Job-Description Fit Scorer

A production-grade, applied-AI fit assessment engine built with **FastAPI**, **Sentence Transformers (MiniLM / BGE)**, and **Multi-Tier LLM Reasoning (Groq & Gemini)** with deterministic rule fallbacks and adversarial prompt injection defenses.

Given a Job Description (plain text or file) and a Candidate Resume (PDF, DOCX, TXT), the engine extracts structured requirements, scores the candidate across **5 explicit dimensions**, enforces **hard disqualification caps**, and generates **actionable, per-criterion recruiter reasoning**.

---

## 🚀 Key Features & Architecture

```
                                 ┌──────────────────────────────┐
                                 │  Incoming Request (/score)   │
                                 │   - Resume (PDF/DOCX/TXT)    │
                                 │   - JD (Text or File upload) │
                                 └──────────────┬───────────────┘
                                                │
                                                ▼
                                 ┌──────────────────────────────┐
                                 │ Safe File & Steganography    │
                                 │ Parser (Strips white/4pt txt)│
                                 └──────────────┬───────────────┘
                                                │
                     ┌──────────────────────────┴──────────────────────────┐
                     ▼                                                     ▼
      ┌─────────────────────────────┐                       ┌─────────────────────────────┐
      │  JD Criteria Extraction     │                       │ Prompt Injection Screening  │
      │  - Tech Skills Vocab (~300) │                       │ - Llama Prompt Guard 2 (86M)│
      │  - Min Experience & Seniority│                       │ - Heuristic Regex Filter    │
      │  - Degree Hierarchy & Quality│                       └──────────────┬──────────────┘
      └──────────────┬──────────────┘                                      │
                     │                                                     │
                     └──────────────────────────┬──────────────────────────┘
                                                ▼
                                 ┌──────────────────────────────┐
                                 │ 5-Dimension Scorer (Config)  │
                                 │ 1. Skills Match (35%)        │
                                 │ 2. Semantic Sim (20%, MiniLM)│
                                 │ 3. Experience Match (25%)    │
                                 │ 4. Education Match (10%)     │
                                 │ 5. Seniority Fit (10%)       │
                                 │ Hard Caps (Skills/Seniority) │
                                 └──────────────┬───────────────┘
                                                │
                                                ▼
                                 ┌──────────────────────────────┐
                                 │ Multi-Tier LLM Reasoner      │
                                 │ Tier 1: Groq (gpt-oss-20b)   │
                                 │ Tier 2: Groq (gpt-oss-120b)  │
                                 │ Tier 3: Gemini 2.0 Flash-Lite│
                                 │ Tier 4: Deterministic Rules  │
                                 └──────────────┬───────────────┘
                                                │
                                                ▼
                                 ┌──────────────────────────────┐
                                 │ Structured Fit Assessment    │
                                 │ - Overall Score (0-100)      │
                                 │ - Per-Criterion Reasoning    │
                                 │ - Latency Telemetry Matrix   │
                                 └──────────────────────────────┘
```

### 1. 5-Dimension Scoring Engine
- **Skills Match (35% default)**: Frequency-weighted keyword analysis (0 hits = 0.0, 1 mention = 0.7, 2+ occurrences = 1.0).
- **Semantic Similarity (20% default)**: Cosine similarity via `all-MiniLM-L6-v2` (with automatic fallback to `BAAI/bge-small-en-v1.5` and neutral 0.5 baseline).
- **Experience Match (25% default)**: Date range detection (e.g. `2021 – 2025`) and explicit duration parsing compared against JD minimum requirement.
- **Education Match (10% default)**: Hierarchical degree mapping (High School $\rightarrow$ Diploma $\rightarrow$ Bachelors $\rightarrow$ Masters $\rightarrow$ PhD) with STEM relevance bonus.
- **Seniority Fit (10% default)**: Title trajectory detection compared against target role level.

### 2. Configurable Weights & Hard Caps (No Hardcoding)
All weights, thresholds, and limits live in `config/weights.yaml`. Weights are dynamically validated and normalized at startup. Hard caps safeguard against inflated scores (e.g., severe seniority mismatch or skills match < 25% capped at 45.0).

### 3. Prompt Injection Defense & Steganography Filter
- **PDF Steganography Filter**: Strips invisible/white-colored text (`#FFFFFF`) and tiny font sizes (< 4pt) often used in adversarial resume manipulation.
- **Llama Prompt Guard 2 (86M)**: Screens text before LLM reasoning. If an injection attempt is flagged, the LLM call is bypassed and deterministic rule reasoning is returned.
- **Score Integrity**: All mathematical scores are computed by deterministic Python functions, making the actual numerical fit score 100% immune to prompt injection.

---

## 📦 Setup & Installation

### Prerequisites
- Python 3.10+ (tested on Python 3.13)
- Git

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/predator279/resume-fit-scorer.git
cd resume-fit-scorer

# Create virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
# source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Environment Variables (Optional)
Copy `.env.example` to `.env` if you want LLM natural-language reasoning:
```bash
cp .env.example .env
```
Add your API keys:
```ini
GROQ_API_KEY=gsk_...
GEMINI_API_KEY=AIzaSy...
```
*(Note: If no API keys are provided, the system automatically runs in **Tier 4 deterministic rule mode** with full scoring and reasoning without failing).*

---

## 🏃 Running the Application

### Start the FastAPI Server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
API Documentation & Interactive Swagger UI:
- **Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check**: [http://localhost:8000/health](http://localhost:8000/health)
- **Active Weights**: [http://localhost:8000/weights](http://localhost:8000/weights)

---

## 📡 API Usage & Examples

### 1. Score Fit (cURL — Text JD + PDF/TXT Resume)
```bash
curl -X POST "http://localhost:8000/score" \
  -F "resume=@samples/resume_sample_1_strong.txt" \
  -F "jd_text=Looking for a Senior Python Backend Engineer with 5+ years experience in FastAPI, PostgreSQL, Docker, AWS, and Microservices. Bachelor's degree required."
```

### 2. Score Fit (cURL — JD as File Upload)
```bash
curl -X POST "http://localhost:8000/score" \
  -F "resume=@samples/resume_sample_1_strong.txt" \
  -F "jd_file=@samples/jd_senior_backend.txt"
```

### 3. Example Response Payload
```json
{
  "success": true,
  "parse_failed": false,
  "overall_score": 88.6,
  "grade": "Strong Match",
  "recruiter_summary": "Strong candidate match. Exceeds core technical and experience requirements; highly recommended for interview.",
  "criteria": [
    {
      "name": "Skills Match",
      "key": "skills_match",
      "weight": 0.35,
      "score": 0.8389,
      "weighted_contribution": 29.36,
      "matched": ["Python", "FastAPI", "PostgreSQL", "Docker", "AWS", "Redis", "Microservices"],
      "missing": ["Distributed Systems", "GraphQL"],
      "reasoning": "Strong skills alignment: matched 16/18 required skills."
    },
    {
      "name": "Semantic Similarity",
      "key": "semantic_similarity",
      "weight": 0.20,
      "score": 0.7864,
      "weighted_contribution": 15.73,
      "reasoning": "High contextual alignment with the overall job scope and domain responsibilities."
    },
    {
      "name": "Experience Match",
      "key": "experience_match",
      "weight": 0.25,
      "score": 1.0,
      "weighted_contribution": 25.0,
      "detected_years": 6.0,
      "required_years": 5,
      "reasoning": "Meets/exceeds experience requirement (6.0 years detected vs 5 years required)."
    },
    {
      "name": "Education Match",
      "key": "education_match",
      "weight": 0.10,
      "score": 1.0,
      "weighted_contribution": 10.0,
      "detected": "Bachelors",
      "required": "Bachelors",
      "reasoning": "Education requirement met: Bachelors (required Bachelors)."
    },
    {
      "name": "Seniority Fit",
      "key": "seniority_fit",
      "weight": 0.10,
      "score": 0.85,
      "weighted_contribution": 8.5,
      "detected": "Principal",
      "required": "Lead",
      "reasoning": "Seniority slightly above target (Principal vs Lead)."
    }
  ],
  "applied_caps": [],
  "parse_metadata": {
    "resume_filename": "resume_sample_1_strong.txt",
    "resume_chars_extracted": 1642,
    "jd_source": "form_text",
    "jd_chars_extracted": 680,
    "jd_quality": "high",
    "prompt_injection_screened": true,
    "prompt_injection_flagged": false,
    "llm_reasoning_provider": "groq:openai/gpt-oss-20b",
    "latency_breakdown": {
      "file_parse_ms": 0.26,
      "jd_extraction_ms": 12.4,
      "prompt_guard_ms": 1.25,
      "embedding_ms": 18.2,
      "rule_scoring_ms": 2.1,
      "llm_reasoning_ms": 684.5,
      "total_ms": 718.71
    }
  }
}
```

---

## 🧪 Calibration & Edge Case Benchmarks

To run the automated verification benchmark across all sample profiles:
```bash
python test_scorer.py
```
To run the full FastAPI integration test suite:
```bash
python test_api.py
```

### Verified Calibration Results (`samples/jd_senior_backend.txt`):
| Profile Sample | Description | Overall Score | Calibration Check vs Sample 1 |
|---|---|---|---|
| **Sample 1** | Senior Backend Engineer (6 yrs exp, 16/18 skills) | **88.6 / 100** | Baseline |
| **Sample 2** | Mid Backend Developer (4 yrs exp, 12/18 skills) | **68.3 / 100** | **Gap = 20.3 pts** (Passes $< 40$ pt constraint) |
| **Sample 3** | Junior Frontend Developer (1 yr exp, 1/18 skills) | **24.6 / 100** | Disqualified via hard cap |
| **Adversarial** | Jailbreak injection attempt | **21.6 / 100** | Flagged by Prompt Guard |
| **Unreadable** | 0-byte or flat scanned image PDF | **`parse_failed: true`** | Handled with HTTP 200 diagnosis |

---

## ⚙️ Configuration (`config/weights.yaml`)

You can modify scoring weights, hard-cap ceilings, or model names at any time without touching Python code:
```yaml
weights:
  skills_match: 0.35
  semantic_similarity: 0.20
  experience_match: 0.25
  education_match: 0.10
  seniority_fit: 0.10

hard_caps:
  low_skills_threshold: 0.25
  low_skills_max_score: 45.0
  seniority_mismatch_max_score: 60.0
```

---

## 🔍 What Works vs. What Doesn't (Honest Account)

### ✅ What Works Robustly:
1. **Multi-format parsing**: Fast, clean text extraction for PDF, DOCX (including embedded table rows), and TXT.
2. **Score Calibration**: Proportional separation between adjacent candidate tiers without non-linear score collapse.
3. **Resilient AI Pipeline**: Multi-model fallback for embeddings (`MiniLM` $\rightarrow$ `BGE` $\rightarrow$ Neutral) and LLM reasoning (`Groq` $\rightarrow$ `Gemini` $\rightarrow$ Rule-based).
4. **Adversarial Safety**: Mathematical scoring cannot be hijacked via prompt injection; hidden text in PDFs is stripped.
5. **Detailed Telemetry**: Microsecond-accurate latency breakdown per sub-component in every response.

### ⚠️ Current Limitations & Roadmap:
1. **Scanned / Flat Image PDFs**: Relies on native digital text streams. Scanned image PDFs return a graceful `parse_failed` error rather than invoking heavy OCR (Tesseract / EasyOCR integration is planned for v2).
2. **Non-English Language Resumes**: MiniLM handles multilingual embeddings, but technical vocabulary regex matching is currently optimized for English JDs and resumes.
3. **Multi-Column PDF Layouts**: Extremely intricate multi-column PDFs can occasionally interleave job experience lines; pdfminer fallback mitigates this, but structural bounding-box reconstruction would further improve accuracy.
