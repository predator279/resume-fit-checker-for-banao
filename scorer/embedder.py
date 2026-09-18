"""
embedder.py — Semantic text embedding and cosine similarity calculation.
Uses sentence-transformers with primary model (all-MiniLM-L6-v2), fallback model (BAAI/bge-small-en-v1.5),
and robust error handling defaulting to a neutral similarity score (0.5) if all fails.
"""

import logging
from typing import Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .config_loader import get_models_config

logger = logging.getLogger(__name__)

_MODEL_INSTANCE = None
_MODEL_NAME_LOADED = None


def get_embedding_model():
    """
    Lazy singleton loader for the sentence transformer embedding model.
    Attempts primary model first, falls back to secondary model on failure.
    """
    global _MODEL_INSTANCE, _MODEL_NAME_LOADED
    if _MODEL_INSTANCE is not None:
        return _MODEL_INSTANCE

    models_cfg = get_models_config()
    primary = models_cfg.get("embedding_primary", "all-MiniLM-L6-v2")
    fallback = models_cfg.get("embedding_fallback", "BAAI/bge-small-en-v1.5")

    try:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading primary embedding model: %s", primary)
        _MODEL_INSTANCE = SentenceTransformer(primary)
        _MODEL_NAME_LOADED = primary
        logger.info("Primary embedding model loaded successfully.")
        return _MODEL_INSTANCE
    except Exception as e:
        logger.warning("Failed to load primary embedding model %s: %s. Attempting fallback %s", primary, e, fallback)

    try:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading fallback embedding model: %s", fallback)
        _MODEL_INSTANCE = SentenceTransformer(fallback)
        _MODEL_NAME_LOADED = fallback
        logger.info("Fallback embedding model loaded successfully.")
        return _MODEL_INSTANCE
    except Exception as e:
        logger.error("Failed to load fallback embedding model %s: %s. Embedder running in neutral mode.", fallback, e)
        _MODEL_INSTANCE = None
        _MODEL_NAME_LOADED = "none"
        return None


def calculate_semantic_similarity(resume_text: str, jd_text: str) -> dict:
    """
    Computes cosine similarity between resume text and JD text.
    Handles truncated length, model inference exceptions, and returns score + metadata.

    Returns:
        {
            "score": float (0.0 to 1.0),
            "model_used": str,
            "fallback_used": bool,
            "error": str | None
        }
    """
    if not resume_text.strip() or not jd_text.strip():
        return {
            "score": 0.5,
            "model_used": "none",
            "fallback_used": True,
            "note": "Empty text provided for semantic match.",
        }

    model = get_embedding_model()
    if model is None:
        return {
            "score": 0.5,
            "model_used": "none",
            "fallback_used": True,
            "note": "Embedding model unavailable — defaulting to neutral score 0.5.",
        }

    try:
        # Truncate to first 3000 chars to avoid memory issues and focus on core content
        r_snippet = resume_text[:3000]
        jd_snippet = jd_text[:3000]

        embeddings = model.encode(
            [r_snippet, jd_snippet],
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        raw_sim = float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])
        # Bound score between 0.0 and 1.0
        normalized_sim = max(0.0, min(1.0, raw_sim))

        return {
            "score": round(normalized_sim, 4),
            "model_used": _MODEL_NAME_LOADED,
            "fallback_used": False,
        }
    except Exception as e:
        logger.exception("Error during embedding inference: %s", e)
        return {
            "score": 0.5,
            "model_used": _MODEL_NAME_LOADED or "none",
            "fallback_used": True,
            "error": str(e),
            "note": "Error in semantic embedding inference — defaulting to neutral score 0.5.",
        }
