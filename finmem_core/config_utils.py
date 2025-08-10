
# finmem_core/config_utils.py
from __future__ import annotations

import os
from typing import Any, Dict

# Paper-grounded defaults
# - Layer stabilities Q_l ≈ {14, 90, 365} and piecewise/decay behavior are described in the paper.
# - We don't reimplement the math here; these defaults simply pre-fill expected config keys so code never KeyErrors.
#
# Notes:
# - Keep compatibility with both 'model' and 'model_name' for embedding (paper used text-embedding-ada-002).
# - Provide safe defaults for the four memory layers expected by the code: short/mid/long/reflection.

_GENERAL_DEFAULTS: Dict[str, Any] = {
    "agent_name": "agent_1",
    "trading_symbol": "AAPL",
    "character_string": "",
    "look_back_window_size": 3,   # M from the paper's testing reflection window (typical small integer)
    "top_k": 5,                   # Top-K events per layer; paper explores K in {1,3,5,10}
    "use_gpu": False,
}

_EMBEDDING_DEFAULTS: Dict[str, Any] = {
    "model_name": "text-embedding-3-small",  # backwards compat below will map older names
    "chunk_char_size": 6000,
}

_LAYER_DEFAULTS: Dict[str, Any] = {
    # thresholds are guardrails for cleanup / promotion logic in local implementation
    "jump_threshold_upper": 999_999_999.0,
    "jump_threshold_lower": -999_999_999.0,
    "importance_score_initialization": "sample",
    "decay_params": {},
    "clean_up_threshold_dict": {"recency_threshold": 0.05, "importance_threshold": 5.0},
}

def _ensure_layer(cfg: Dict[str, Any], key: str) -> Dict[str, Any]:
    layer = dict(_LAYERS_TEMPLATE[key])  # copy template
    user = cfg.get(key, {})
    if isinstance(user, dict):
        layer.update(user)
    return layer

# Each layer gets same structural defaults; you can specialize later in UI if desired.
_LAYERS_TEMPLATE = {
    "short": dict(_LAYER_DEFAULTS),
    "mid": dict(_LAYER_DEFAULTS),
    "long": dict(_LAYER_DEFAULTS),
    "reflection": dict(_LAYER_DEFAULTS),
}

def _backcompat_embedding_names(emb: Dict[str, Any]) -> Dict[str, Any]:
    # Accept 'model' or 'model_name'; prefer 'model_name' in the normalized config.
    model = emb.get("model_name") or emb.get("model")
    if not model:
        model = _EMBEDDING_DEFAULTS["model_name"]
    # Map paper-era default to a current one if user didn't explicitly choose another
    if model == "text-embedding-ada-002":
        # preserve user's explicit choice; otherwise, allow UI to still use ada-002 if wanted
        # we'll just normalize the key name
        pass
    emb_norm = {
        "model_name": model,
        "chunk_char_size": int(emb.get("chunk_char_size", _EMBEDDING_DEFAULTS["chunk_char_size"])),
    }
    return emb_norm

def _ensure_section(cfg: Dict[str, Any], name: str, defaults: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(defaults)
    user = cfg.get(name, {})
    if isinstance(user, dict):
        out.update(user)
    return out

def normalize_and_validate_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    if cfg is None:
        cfg = {}

    # General
    cfg["general"] = _ensure_section(cfg, "general", _GENERAL_DEFAULTS)

    # Embedding
    emb_in = cfg.get("embedding", {})
    cfg["embedding"] = _backcompat_embedding_names(emb_in)

    # Chat block is free-form; just ensure it exists to avoid KeyError in callers
    cfg["chat"] = dict(cfg.get("chat") or {})

    # Memory layers
    cfg["short"] = _ensure_layer(cfg, "short")
    cfg["mid"] = _ensure_layer(cfg, "mid")
    cfg["long"] = _ensure_layer(cfg, "long")
    cfg["reflection"] = _ensure_layer(cfg, "reflection")

    # Friendly validations (non-fatal)
    problems = []

    # OpenAI key check (info only – do not crash)
    if not (os.getenv("OPENAI_API_KEY") or cfg["chat"].get("api_key")):
        problems.append("OPENAI_API_KEY is not set; only local/tokenizer paths would work.")

    # Embedding model sanity
    m = (cfg["embedding"].get("model_name") or "").strip()
    if not m:
        problems.append("embedding.model_name is empty; defaulting to text-embedding-3-small.")
        cfg["embedding"]["model_name"] = _EMBEDDING_DEFAULTS["model_name"]

    # General sanity
    if cfg["general"].get("top_k", 0) < 0:
        problems.append("general.top_k was negative; reset to 0.")
        cfg["general"]["top_k"] = 0

    # Create a diagnostics field the app can read & display
    cfg["_normalized_warnings"] = problems
    return cfg
