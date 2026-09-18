"""
jd_parser.py — Auto-extract structured criteria from raw Job Description text.

Returns:
  skills: list of tech/domain skills found in the JD
  experience_years: minimum years of experience required (int or None)
  education: highest required degree label (str or None)
  seniority: "Junior" | "Mid" | "Senior" | "Lead" | "Principal" | None
  jd_quality: "high" | "medium" | "low"
"""

import re
from typing import Dict, Any, List, Optional

# Curated tech / domain vocabulary (~300 terms) sorted longest-first
TECH_VOCAB = sorted([
    # Languages
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "golang",
    "rust", "kotlin", "swift", "scala", "ruby", "php", "r", "matlab", "julia",
    "perl", "bash", "shell", "powershell", "groovy", "sql",
    # Web & Frameworks
    "html", "css", "react", "react.js", "reactjs", "vue", "vue.js", "vuejs",
    "angular", "next.js", "nextjs", "nuxt", "svelte", "node.js", "nodejs",
    "express", "fastapi", "flask", "django", "spring", "spring boot",
    "asp.net", ".net", "rails", "ruby on rails", "laravel", "graphql", "rest", "restful",
    "websocket", "webpack", "vite", "tailwind", "tailwind css", "bootstrap", "sass",
    # Data / ML / AI
    "machine learning", "deep learning", "natural language processing", "nlp",
    "computer vision", "cv", "reinforcement learning", "data science",
    "data analysis", "data engineering", "feature engineering",
    "tensorflow", "pytorch", "keras", "scikit-learn", "sklearn",
    "huggingface", "transformers", "bert", "gpt", "llm", "rag",
    "langchain", "llamaindex", "openai", "anthropic", "xgboost", "lightgbm",
    "pandas", "numpy", "scipy", "matplotlib", "seaborn", "plotly",
    "opencv", "pillow",
    # Databases & Storage
    "mysql", "postgresql", "postgres", "sqlite", "oracle", "sql server",
    "mongodb", "cassandra", "redis", "elasticsearch", "dynamodb",
    "neo4j", "influxdb", "clickhouse", "snowflake", "redshift", "bigquery",
    "hive", "hbase", "couchdb", "firebase", "supabase",
    # Cloud & DevOps
    "aws", "azure", "gcp", "google cloud",
    "ec2", "s3", "lambda", "ecs", "eks", "fargate", "rds", "sqs", "sns",
    "docker", "kubernetes", "k8s", "helm", "terraform", "ansible",
    "jenkins", "github actions", "gitlab ci", "circleci", "travis ci",
    "ci/cd", "devops", "sre", "linux", "unix", "nginx", "apache",
    "prometheus", "grafana", "datadog", "splunk",
    # Data pipeline & Big Data
    "spark", "apache spark", "hadoop", "kafka", "apache kafka", "flink",
    "airflow", "apache airflow", "dbt", "prefect", "luigi", "databricks",
    # Tools & Practices
    "git", "github", "gitlab", "bitbucket", "jira", "confluence",
    "agile", "scrum", "kanban", "tdd", "bdd", "microservices",
    "api", "apis", "sdk", "orm", "jwt", "oauth", "grpc", "protobuf",
    # Mobile
    "android", "ios", "react native", "flutter", "xamarin",
    # Security
    "cybersecurity", "penetration testing", "owasp", "siem", "iam",
    # BI / Analytics
    "tableau", "power bi", "looker", "qlik", "dax", "excel",
    # System Design & Architecture
    "system design", "distributed systems", "load balancing", "caching",
], key=len, reverse=True)

SENIORITY_MAP = {
    "intern": "intern",
    "internship": "intern",
    "fresher": "junior",
    "entry level": "junior",
    "entry-level": "junior",
    "junior": "junior",
    "associate": "junior",
    "mid level": "mid",
    "mid-level": "mid",
    "mid senior": "senior",
    "senior": "senior",
    "sr.": "senior",
    "sr ": "senior",
    "staff": "senior",
    "lead": "lead",
    "tech lead": "lead",
    "principal": "principal",
    "architect": "principal",
    "manager": "lead",
    "director": "principal",
    "head of": "principal",
    "vp": "principal",
}

SENIORITY_ORDER = ["intern", "junior", "mid", "senior", "lead", "principal"]

DEGREE_LEVELS = {
    "phd": 5, "ph.d": 5, "doctorate": 5, "doctoral": 5,
    "masters": 4, "master": 4, "m.tech": 4, "m.s": 4, "mtech": 4,
    "m.e": 4, "mba": 4, "m.sc": 4, "msc": 4, "m.b.a": 4,
    "bachelors": 3, "bachelor": 3, "b.tech": 3, "b.e": 3, "btech": 3,
    "b.sc": 3, "bsc": 3, "b.s": 3, "undergraduate": 3,
    "diploma": 2, "polytechnic": 2,
    "12th": 1, "hsc": 1, "high school": 1,
}

DEGREE_LABEL = {
    5: "PhD / Doctorate",
    4: "Masters",
    3: "Bachelors",
    2: "Diploma",
    1: "High School",
}


def parse_jd(jd_text: str) -> Dict[str, Any]:
    """
    Parse a free-form job description string and extract structured criteria.

    Returns:
        {
            "skills": list[str],
            "experience_years": int | None,
            "education": str | None,
            "seniority": str | None,
            "jd_quality": "high" | "medium" | "low"
        }
    """
    if not jd_text or not jd_text.strip():
        return {
            "skills": [],
            "experience_years": None,
            "education": None,
            "seniority": None,
            "jd_quality": "low",
        }

    text_lower = jd_text.lower()
    skills = _extract_skills(jd_text, text_lower)
    experience_years = _extract_experience(text_lower)
    education = _extract_education(text_lower)
    seniority = _extract_seniority(text_lower)

    # Evaluate JD quality
    if len(skills) >= 4 and (experience_years is not None or seniority is not None):
        quality = "high"
    elif len(skills) >= 1 or experience_years is not None:
        quality = "medium"
    else:
        quality = "low"

    return {
        "skills": skills,
        "experience_years": experience_years,
        "education": education,
        "seniority": seniority,
        "jd_quality": quality,
    }


def _extract_skills(original: str, text_lower: str) -> List[str]:
    found: List[str] = []
    consumed_spans = []

    for term in TECH_VOCAB:
        pattern = r"(?<![a-zA-Z0-9+#.])" + re.escape(term) + r"(?![a-zA-Z0-9+#.])"
        for m in re.finditer(pattern, text_lower):
            start, end = m.start(), m.end()
            if any(cs <= start and ce >= end for cs, ce in consumed_spans):
                continue
            consumed_spans.append((start, end))
            found.append(_pretty_case(term))
            break

    seen = set()
    unique = []
    for s in found:
        key = s.lower()
        if key not in seen:
            seen.add(key)
            unique.append(s)

    return unique


def _pretty_case(term: str) -> str:
    ALWAYS_UPPER = {
        "sql", "html", "css", "api", "apis", "aws", "gcp", "ux", "ui", "nlp", "cv",
        "tdd", "bdd", "sdk", "orm", "jwt", "sre", "iam", "ci/cd", "rest", "restful",
        "grpc", "vp", "hsc", "php", "r", "k8s",
    }
    ALWAYS_AS_IS = {
        "python": "Python", "javascript": "JavaScript", "typescript": "TypeScript",
        "c++": "C++", "c#": "C#", "go": "Go", "rust": "Rust",
        "react.js": "React.js", "reactjs": "React", "vue.js": "Vue.js",
        "next.js": "Next.js", "node.js": "Node.js", "asp.net": "ASP.NET",
        ".net": ".NET", "tensorflow": "TensorFlow", "pytorch": "PyTorch",
        "scikit-learn": "Scikit-learn", "huggingface": "HuggingFace",
        "langchain": "LangChain", "llamaindex": "LlamaIndex", "openai": "OpenAI",
        "xgboost": "XGBoost", "lightgbm": "LightGBM", "mongodb": "MongoDB",
        "postgresql": "PostgreSQL", "elasticsearch": "Elasticsearch",
        "dynamodb": "DynamoDB", "clickhouse": "ClickHouse", "github": "GitHub",
        "gitlab": "GitLab", "bitbucket": "Bitbucket", "jenkins": "Jenkins",
        "kubernetes": "Kubernetes", "terraform": "Terraform", "ansible": "Ansible",
        "prometheus": "Prometheus", "grafana": "Grafana", "datadog": "Datadog",
        "databricks": "Databricks", "kafka": "Kafka", "airflow": "Airflow",
        "github actions": "GitHub Actions", "gitlab ci": "GitLab CI",
        "circleci": "CircleCI", "spring boot": "Spring Boot", "fastapi": "FastAPI",
        "graphql": "GraphQL", "websocket": "WebSocket",
        "machine learning": "Machine Learning", "deep learning": "Deep Learning",
        "natural language processing": "NLP", "computer vision": "Computer Vision",
        "data science": "Data Science", "data engineering": "Data Engineering",
        "react native": "React Native", "power bi": "Power BI",
        "apache spark": "Apache Spark", "apache kafka": "Apache Kafka",
        "apache airflow": "Apache Airflow", "tailwind css": "Tailwind CSS",
        "ruby on rails": "Ruby on Rails", "system design": "System Design",
        "distributed systems": "Distributed Systems",
    }
    t = term.lower()
    if t in ALWAYS_AS_IS:
        return ALWAYS_AS_IS[t]
    if t in ALWAYS_UPPER:
        return t.upper()
    return " ".join(w.capitalize() for w in term.split())


def _extract_experience(text_lower: str) -> Optional[int]:
    candidates = []

    for m in re.finditer(r"(\d+)\s*[-–]\s*(\d+)\s*(?:\+?\s*)?years?", text_lower):
        candidates.append(int(m.group(1)))

    patterns = [
        r"(\d+)\s*\+\s*years?",
        r"minimum\s+of\s+(\d+)\s+years?",
        r"minimum\s+(\d+)\s+years?",
        r"at\s+least\s+(\d+)\s+years?",
        r"(\d+)\s+years?\s+of\s+(?:experience|work)",
        r"(\d+)\s+years?\s+(?:relevant|professional|industry)",
        r"(\d+)\+?\s+yrs?",
    ]
    for pat in patterns:
        for m in re.finditer(pat, text_lower):
            val = int(m.group(1))
            if 0 < val < 40:
                candidates.append(val)

    if not candidates:
        return None
    return min(candidates)


def _extract_education(text_lower: str) -> Optional[str]:
    best_level = 0
    for keyword, level in DEGREE_LEVELS.items():
        pattern = r"(?<![a-z])" + re.escape(keyword) + r"(?![a-z])"
        if re.search(pattern, text_lower) and level > best_level:
            best_level = level
    return DEGREE_LABEL.get(best_level)


def _extract_seniority(text_lower: str) -> Optional[str]:
    detected = []
    for keyword, level in SENIORITY_MAP.items():
        if keyword in text_lower:
            detected.append(level)

    if not detected:
        return None

    for lvl in reversed(SENIORITY_ORDER):
        if lvl in detected:
            return lvl.capitalize()

    return detected[-1].capitalize()
