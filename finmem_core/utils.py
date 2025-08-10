import os, re, json, time, datetime as dt
from typing import Any, Dict, Optional

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def stamp_dir(base: str) -> str:
    ensure_dir(base)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(base, ts)
    ensure_dir(out)
    return out

def load_json(path: str, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default

def save_json(path: str, data: Any):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def print_warnings(warnings):
    if not warnings:
        return ""
    return "\n".join(f"⚠️ {w}" for w in warnings)

def safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default
