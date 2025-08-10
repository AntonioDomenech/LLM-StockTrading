# finmem_core/checkpoint_utils.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, List


def _env_dir_for_final(final: Path) -> Optional[Path]:
    """
    Return the directory that contains env.pkl for a given final/ dir.
    Supports both:
      - final/env/env.pkl
      - final/env/env/env.pkl   (extra nested 'env')
    """
    cand1 = final / "env" / "env.pkl"
    if cand1.exists():
        return cand1.parent
    cand2 = final / "env" / "env" / "env.pkl"
    if cand2.exists():
        return cand2.parent
    return None


def _agent_dir_for_final(final: Path) -> Optional[Path]:
    """
    Choose the agent dir that contains 'brain/'.
    Prefer FLAT layout if present and accompanied by 'state_dict.pkl':
      - final/state_dict.pkl
      - final/brain/
    Otherwise fall back to NESTED layout:
      - final/<agent_name>/state_dict.pkl
      - final/<agent_name>/brain/
    Returns the *agent dir* (parent of 'brain').
    """
    flat_agent_dir = final if (final / "brain").exists() else None
    nested_brains = sorted(final.glob("*/brain"),
                           key=lambda p: p.stat().st_mtime,
                           reverse=True)
    nested_agent_dir = nested_brains[0].parent if nested_brains else None

    if flat_agent_dir and (flat_agent_dir / "state_dict.pkl").exists():
        return flat_agent_dir
    if nested_agent_dir and (nested_agent_dir / "state_dict.pkl").exists():
        return nested_agent_dir
    if flat_agent_dir:
        return flat_agent_dir
    if nested_agent_dir:
        return nested_agent_dir
    return None


def _is_valid_final_dir(final: Path) -> bool:
    env_dir = _env_dir_for_final(final)
    agent_dir = _agent_dir_for_final(final)
    # require env and agent dirs; allow agent without state_dict.pkl but prefer with
    return (env_dir is not None) and (agent_dir is not None)


def list_run_candidates(base: str = "data/runs") -> List[Dict[str, str]]:
    base_p = Path(base)
    if not base_p.exists():
        return []

    candidates: List[Dict[str, str]] = []

    # Expected: data/runs/<stamp>/<run_name>/final/
    for final in base_p.glob("*/*/final"):
        if not final.is_dir():
            continue
        if not _is_valid_final_dir(final):
            continue
        env_dir = _env_dir_for_final(final)
        agent_dir = _agent_dir_for_final(final)
        candidates.append({
            "final": str(final),
            "env_dir": str(env_dir),     # type: ignore[arg-type]
            "agent_dir": str(agent_dir), # type: ignore[arg-type]
            "mtime": final.stat().st_mtime,
        })

    # Also support flat: data/runs/<stamp>/final
    for final in base_p.glob("*/final"):
        if not final.is_dir():
            continue
        if not _is_valid_final_dir(final):
            continue
        env_dir = _env_dir_for_final(final)
        agent_dir = _agent_dir_for_final(final)
        candidates.append({
            "final": str(final),
            "env_dir": str(env_dir),     # type: ignore[arg-type]
            "agent_dir": str(agent_dir), # type: ignore[arg-type]
            "mtime": final.stat().st_mtime,
        })

    candidates.sort(key=lambda x: x["mtime"], reverse=True)
    return candidates


def find_latest_final(base: str = "data/runs") -> Optional[Dict[str, str]]:
    cands = list_run_candidates(base=base)
    return cands[0] if cands else None


def resolve_test_checkpoint(
    base: str = "data/runs",
    prefer_final_dir: Optional[str] = None
) -> tuple[str, str]:
    """
    Returns (env_dir, agent_dir).
    - If prefer_final_dir is provided, validate and use that final/.
    - Otherwise pick the newest valid final/ under `base`.
    Raises FileNotFoundError if none.
    """
    if prefer_final_dir:
        final = Path(prefer_final_dir)
        if not final.exists():
            raise FileNotFoundError(f"{prefer_final_dir!r} does not exist")
        env_dir = _env_dir_for_final(final)
        agent_dir = _agent_dir_for_final(final)
        if not env_dir or not agent_dir:
            raise FileNotFoundError(f"Invalid final dir: {prefer_final_dir}")
        return str(env_dir), str(agent_dir)

    latest = find_latest_final(base=base)
    if not latest:
        raise FileNotFoundError(f"No TRAIN checkpoint found under {base}.")
    return latest["env_dir"], latest["agent_dir"]
