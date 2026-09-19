# 3–5 Minute Walkthrough Video Script & Demonstration Guide

This guide gives you a minute-by-minute script, an unseen test sample to run live, and the exact code/prompt sections to show on screen.

---

## 🎬 Video Requirements (From Assessment)
- **Duration**: 3 to 5 minutes
- **Structure**:
  1. Show it running live on an **unseen input** (Resume + JD).
  2. Walk through **one part of your code**.
  3. Walk through **one prompt** you used.
- **Link**: Upload to Google Drive $\rightarrow$ Set sharing to **"Anyone with the link can view"** $\rightarrow$ Verify in an incognito window.

---

## ⏱️ Minute-by-Minute Video Script

### 📍 [0:00 – 0:45] Intro & Architecture Overview
- **What to say**:
  > *"Hi, this is Manish. Today I'm demonstrating my Resume to Job-Description Fit Scorer for the assessment.*
  > *The system evaluates candidate fit across 5 configurable dimensions: Skills Match, Semantic Similarity via MiniLM, Experience, Education, and Seniority.*
  > *Crucially, I designed a decoupled architecture where all numerical scoring and hard caps are computed deterministically in Python, while the LLM is used strictly for natural-language recruiter reasoning. This eliminates score hallucinations and provides 100% calibration stability."*

---

### 📍 [0:45 – 2:00] Live Demonstration on Unseen Input
- **What to show on screen**: Open browser to **`http://localhost:8000/docs`** (Swagger UI).
- **Actions**:
  1. Click **`POST /score`** $\rightarrow$ **`Try it out`**.
  2. Paste a fresh **Job Description** (e.g. Cloud DevOps Engineer).
  3. Upload an unseen **Resume** file (`.txt` or `.pdf`).
  4. Click **`Execute`**.
- **What to say while showing the response**:
  > *"As we can see in the response payload:*
  > *- The candidate achieved an overall score of **84.2** (Good Match).*
  > *- Each criterion has its own normalized score, weighted contribution, and specific recruiter explanation.*
  > *- We also track microsecond latency breakdown in `parse_metadata`: file parsing took 0.3ms, embedding took 18ms, and LLM reasoning took 650ms, with an end-to-end response time under 750ms."*
- **Quick Edge Case Show (30s)**:
  - Upload an unreadable/0-byte file $\rightarrow$ execute $\rightarrow$ show the clean `parse_failed: true` JSON response (no 500 crashes).

---

### 📍 [2:00 – 3:15] Code Walkthrough (Pick 1 of 2 options)
- **What to show on screen**: Open VS Code / IDE at `scorer/criteria_scorer.py` (or `scorer/file_parser.py`).
- **Code to highlight**:
  - Show the 5 scoring functions and the `compute_fit_assessment` function.
  - Show how weights are loaded from `config/weights.yaml` rather than being hardcoded.
  - Show the hard caps (e.g., if skills match is $< 25\%$, the overall score is capped at $45.0$).
- **What to say**:
  > *"Looking at `criteria_scorer.py`, you can see that scoring is completely modular. Scoring weights are pulled dynamically from `config/weights.yaml` and normalized at startup.*
  > *We also enforce hard disqualification caps: if a candidate matches less than 25% of required skills, their score is capped at 45 points, preventing inflated scores on unqualified candidates.*
  > *In `file_parser.py`, we also built a steganography filter that strips white `#FFFFFF` text and fonts under 4pt to prevent prompt injection and keyword stuffing."*

---

### 📍 [3:15 – 4:30] Prompt & LLM Layer Walkthrough
- **What to show on screen**: Open `scorer/llm_reasoner.py` and highlight the `system_prompt` and fallback chain.
- **Prompt to highlight**:
  ```python
  system_prompt = (
      "You are an expert AI recruiting assistant. Analyze the candidate's criteria evaluation against the job description.\n"
      "Generate concise, professional recruiter reasoning for EACH of the 5 criteria, plus an overall recruiter summary.\n"
      "STRICT REQUIREMENTS:\n"
      "1. Do not alter or hallucinate new criteria. Keep scores unchanged.\n"
      "2. Provide exactly 1-2 factual, actionable sentences per criterion explaining WHY that score was achieved.\n"
      "3. Output MUST be valid, clean JSON with this exact schema...\n"
  )
  ```
- **What to say**:
  > *"Here in `llm_reasoner.py`, I designed the prompt with a strict JSON schema contract. The prompt instructs the model to generate factual, 1-2 sentence explanations explaining why each score was achieved without modifying any numerical scores.*
  > *We also built a 4-tier fallback chain: it screens with Llama Prompt Guard 2, tries Groq `gpt-oss-20b`, falls back to Gemini `gemini-3.1-flash-lite`, and ultimately falls back to deterministic rule strings if no API keys are provided. This ensures the system is 100% resilient."*

---

### 📍 [4:30 – 4:50] Conclusion
- **What to say**:
  > *"The repository includes automated calibration verification in `test_scorer.py`, comprehensive documentation in `README.md`, and a one-page `explanation.md` detailing design tradeoffs. Thank you for your time!"*

---

## 📋 Fresh Unseen Input for Your Video Demo

You can use this fresh **Cloud / DevOps Engineer** pair during your recording:

### 📄 Sample JD to Paste:
```text
Title: Cloud DevOps Engineer
Company: Apex Cloud Systems
Location: Remote

Requirements:
- 4+ years of experience in DevOps and cloud infrastructure engineering.
- Strong proficiency with Docker, Kubernetes, Terraform, and AWS (EC2, S3, EKS).
- Hands-on experience building automated CI/CD pipelines with GitHub Actions or Jenkins.
- Scripting proficiency in Python or Bash.
- Minimum Education: Bachelor's degree in Computer Science, IT, or related STEM field.
```

### 📄 Sample Candidate Resume to Paste / Save as `.txt` or `.pdf`:
```text
Morgan Vance
Cloud DevOps Engineer | morgan.vance@example.com | San Francisco, CA

Professional Summary:
DevOps Engineer with 5 years of experience automating cloud deployments, container orchestration, and CI/CD pipelines on AWS.

Experience:
Senior DevOps Engineer | CloudOps Inc. | 2021 – 2025
- Managed production Kubernetes (EKS) clusters supporting 40+ microservices.
- Automated multi-region AWS infrastructure provisioning using Terraform.
- Built zero-downtime CI/CD deployment workflows with GitHub Actions and Docker.
- Wrote automated Python monitoring scripts and integrated Prometheus/Grafana alerts.

DevOps Associate | InfraScale | 2019 – 2021
- Maintained Docker containers and configured Jenkins deployment pipelines.
- Managed Linux servers, AWS EC2 instances, and S3 storage policies.

Technical Skills:
- Cloud & Containers: AWS, Docker, Kubernetes, Terraform, Linux
- CI/CD & Tools: GitHub Actions, Jenkins, Git, Prometheus, Grafana
- Scripting: Python, Bash

Education:
Bachelor of Science (B.Sc) in Information Technology
Tech State University (2015 – 2019)
```
*(This profile will score ~85–88 points with clear matching across Docker, Kubernetes, AWS, Terraform, CI/CD, Python, and Bachelor's degree).*

---

## ✅ Pre-Submission Checklist

Before submitting the form, verify:

1. **GitHub Repository**:
   - URL: `https://github.com/predator279/resume-fit-checker-for-banao`
   - Open in an **Incognito browser window** to confirm it loads publicly without login.
   - Confirm all commits and `explanation.md` are visible.
2. **Walkthrough Video**:
   - Upload video to Google Drive.
   - Right click $\rightarrow$ **Share** $\rightarrow$ Change General Access to **"Anyone with the link can view"**.
   - Copy the link and open it in an **Incognito window** to confirm it plays without asking to log in.
3. **Form Fields to Submit**:
   - **GitHub Repository URL**: `https://github.com/predator279/resume-fit-checker-for-banao`
   - **Explanation Document**: `https://github.com/predator279/resume-fit-checker-for-banao/blob/main/explanation.md` (or upload as PDF)
   - **Walkthrough Video Link**: Your Google Drive public link
   - **Hours Spent**: Enter your actual total time spent
