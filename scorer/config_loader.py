"""
config_loader.py — Loads, validates, and normalizes scoring weights and system configuration.
Ensures zero hardcoded weights in scoring logic.
"""

import logging
from pathlib import Path
from typing import Dict, Any
import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG: Dict[str, Any] = {
    "weights": {
        "skills_match": 0.35,
        "semantic_similarity": 0.20,
        "experience_match": 0.25,
        "education_match": 0.10,
        "seniority_fit": 0.10,
    },
    "hard_caps": {
        "low_skills_threshold": 0.25,
        "low_skills_max_score": 45.0,
        "seniority_mismatch_max_score": 60.0,
    },
    "models": {
        "embedding_primary": "all-MiniLM-L6-v2",
        "embedding_fallback": "BAAI/bge-small-en-v1.5",
        "groq_primary_llm": "openai/gpt-oss-20b",
        "groq_secondary_llm": "openai/gpt-oss-120b",
        "groq_guard_model": "meta-llama/llama-prompt-guard-2-86m",
        "gemini_fallback_model": "gemini-2.0-flash-lite",
    },
    "llm": {
        "temperature": 0.0,
        "max_tokens": 600,
        "timeout_seconds": 15,
        "retry_attempts": 2,
    },
    "limits": {
        "max_file_size_mb": 5,
        "min_text_length": 50,
        "max_resume_chars_llm": 4000,
        "max_jd_chars_llm": 2000,
    },
}

_CONFIG_CACHE: Dict[str, Any] = {}


def load_config(config_path: Path | str | None = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file or return cached / default configuration.
    Validates that weights exist and normalizes them to sum precisely to 1.0.
    """
    global _CONFIG_CACHE
    if _CONFIG_CACHE:
        return _CONFIG_CACHE

    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "weights.yaml"
    else:
        config_path = Path(config_path)

    loaded_cfg: Dict[str, Any] = {}
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                loaded_cfg = yaml.safe_load(f) or {}
            logger.info("Loaded configuration from %s", config_path)
        except Exception as e:
            logger.warning("Failed to load %s: %s. Using default config.", config_path, e)
            loaded_cfg = DEFAULT_CONFIG
    else:
        logger.warning("Config file not found at %s. Using default config.", config_path)
        loaded_cfg = DEFAULT_CONFIG

    # Merge with defaults for missing keys
    config = _deep_merge(DEFAULT_CONFIG, loaded_cfg)

    # Normalize weights
    raw_weights = config.get("weights", {})
    total_weight = sum(raw_weights.values())
    if total_weight <= 0:
        logger.error("Invalid total weight sum %s <= 0. Resetting to defaults.", total_weight)
        config["weights"] = DEFAULT_CONFIG["weights"].copy()
    elif abs(total_weight - 1.0) > 0.001:
        logger.warning("Weights sum to %s instead of 1.0. Auto-normalizing.", round(total_weight, 4))
        config["weights"] = {
            k: round(v / total_weight, 4) for k, v in raw_weights.items()
        }

    _CONFIG_CACHE = config
    return _CONFIG_CACHE


def get_weights() -> Dict[str, float]:
    """Return normalized scoring weights dictionary."""
    return load_config().get("weights", DEFAULT_CONFIG["weights"])


def get_hard_caps() -> Dict[str, float]:
    """Return hard cap thresholds and scores."""
    return load_config().get("hard_caps", DEFAULT_CONFIG["hard_caps"])


def get_models_config() -> Dict[str, str]:
    """Return model identifiers."""
    return load_config().get("models", DEFAULT_CONFIG["models"])


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = base.copy()
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged
