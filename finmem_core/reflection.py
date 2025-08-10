from __future__ import annotations
from typing import List, Dict, Union, Any, Optional, Literal
from datetime import date
import json, logging, re
try:
    from rich import print
except Exception:
    from builtins import print
from pydantic import BaseModel, Field, ValidationError
from httpx import HTTPStatusError
from .run_type import RunMode
from .chat import LongerThanContextError
# Try both layouts for prompts
try:
    from .prompts import (
        short_memory_id_desc, mid_memory_id_desc, long_memory_id_desc, reflection_memory_id_desc,
        train_investment_info_prefix, test_investment_info_prefix, test_sentiment_explanation, test_momentum_explanation,
    )
except Exception:
    from prompts.prompts import (
        short_memory_id_desc, mid_memory_id_desc, long_memory_id_desc, reflection_memory_id_desc,
        train_investment_info_prefix, test_investment_info_prefix, test_sentiment_explanation, test_momentum_explanation,
    )

class TrainReflectionOut(BaseModel):
    summary_reason: str = Field("", description="Why the label action makes sense.")
    short_memory_index: int = Field(-1); mid_memory_index: int = Field(-1)
    long_memory_index: int = Field(-1); reflection_memory_index: int = Field(-1)

class TestReflectionOut(BaseModel):
    investment_decision: Literal["buy","sell","hold"] = Field("hold")
    summary_reason: str = Field("")
    short_memory_index: int = Field(-1); mid_memory_index: int = Field(-1)
    long_memory_index: int = Field(-1); reflection_memory_index: int = Field(-1)

def _fmt_mem_section(title: str, items: List[str], ids: List[int]) -> str:
    if not items:
        return f"{title}:\n  (none)\n"
    lines = [f"{title}:"]
    for t, i in zip(items, ids):
        lines.append(f"- ID={i}: {t}")
    return "\n".join(lines) + "\n"

def _build_train_prompt(cur_date: date, symbol: str, future_record: Optional[float],
                        short_memory: List[str], short_ids: List[int],
                        mid_memory: List[str], mid_ids: List[int],
                        long_memory: List[str], long_ids: List[int],
                        reflection_memory: List[str], refl_ids: List[int]) -> str:
    header = train_investment_info_prefix.format(cur_date=cur_date, symbol=symbol, future_record=future_record)
    mem = "\n".join([
        _fmt_mem_section(f"Short-term memory ({short_memory_id_desc})", short_memory, short_ids),
        _fmt_mem_section(f"Mid-term memory ({mid_memory_id_desc})", mid_memory, mid_ids),
        _fmt_mem_section(f"Long-term memory ({long_memory_id_desc})", long_memory, long_ids),
        _fmt_mem_section(f"Reflection memory ({reflection_memory_id_desc})", reflection_memory, refl_ids),
    ])
    schema = ('Respond ONLY in JSON with keys: '
              '{"summary_reason": string, '
              '"short_memory_index": number, "mid_memory_index": number, '
              '"long_memory_index": number, "reflection_memory_index": number}')
    instr = ("Task: Explain briefly the reason for the supervised (label) action and reference the most influential memory IDs. "
             "Pick at most one ID per memory group (use -1 if none).")
    return f"{header}\n{mem}\n{instr}\n{schema}"

def _build_test_prompt(cur_date: date, symbol: str, momentum: Optional[float],
                       short_memory: List[str], short_ids: List[int],
                       mid_memory: List[str], mid_ids: List[int],
                       long_memory: List[str], long_ids: List[int],
                       reflection_memory: List[str], refl_ids: List[int]) -> str:
    header = test_investment_info_prefix.format(cur_date=cur_date, symbol=symbol)
    mem = "\n".join([
        _fmt_mem_section(f"Short-term memory ({short_memory_id_desc})", short_memory, short_ids),
        _fmt_mem_section(f"Mid-term memory ({mid_memory_id_desc})", mid_memory, mid_ids),
        _fmt_mem_section(f"Long-term memory ({long_memory_id_desc})", long_memory, long_ids),
        _fmt_mem_section(f"Reflection memory ({reflection_memory_id_desc})", reflection_memory, refl_ids),
    ])
    helper = (
        "Context:\n"
        f"{test_sentiment_explanation}\n"
        f"{test_momentum_explanation}\n"
        "Momentum shown above is the recent trend proxy.\n"
    )
    schema = ('Respond ONLY in JSON with keys: '
                  '{"investment_decision": "buy" | "sell" | "hold", '
                  '"summary_reason": string, '
                  '"short_memory_index": number, "mid_memory_index": number, '
                  '"long_memory_index": number, "reflection_memory_index": number}')
    instr = ("Task: Make a daily trading decision and justify it briefly. "
             "Choose exactly one of: buy, sell, hold. Reference the most influential memory IDs (use -1 if none).")
    momentum_line = f"Momentum indicator (last days): {momentum}\n"
    return f"{header}\n{momentum_line}\n{mem}\n{helper}\n{instr}\n{schema}"

def _extract_json(text: str) -> Dict[str, Any]:
    if not isinstance(text, str): return {}
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m: return {}
    raw = m.group(0)
    try: return json.loads(raw)
    except Exception:
        raw2 = re.sub(r",\s*\}", "}", raw); raw2 = re.sub(r",\s*\]", "]", raw2)
        try: return json.loads(raw2)
        except Exception: return {}

def trading_reflection(cur_date: date, symbol: str, run_mode: RunMode, endpoint_func,
                       short_memory: List[str], short_memory_id: List[int],
                       mid_memory: List[str], mid_memory_id: List[int],
                       long_memory: List[str], long_memory_id: List[int],
                       reflection_memory: List[str], reflection_memory_id: List[int],
                       future_record: Union[float, None] = None, momentum: Union[float, None] = None,
                       logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    logger = logger or logging.getLogger(__name__)
    if run_mode == RunMode.Train:
        prompt = _build_train_prompt(cur_date, symbol, future_record,
                                     short_memory, short_memory_id, mid_memory, mid_memory_id,
                                     long_memory, long_memory_id, reflection_memory, reflection_memory_id)
    else:
        prompt = _build_test_prompt(cur_date, symbol, momentum,
                                    short_memory, short_memory_id, mid_memory, mid_memory_id,
                                    long_memory, long_memory_id, reflection_memory, reflection_memory_id)
    try:
        raw = endpoint_func(prompt)
    except LongerThanContextError:
        return {"summary_reason": "", "short_memory_index": -1, "mid_memory_index": -1,
                "long_memory_index": -1, "reflection_memory_index": -1} if run_mode==RunMode.Train else                {"investment_decision": "hold", "summary_reason": "", "short_memory_index": -1,
                "mid_memory_index": -1, "long_memory_index": -1, "reflection_memory_index": -1}
    except Exception as e:
        return {"summary_reason": f"(error: {e})", "short_memory_index": -1, "mid_memory_index": -1,
                "long_memory_index": -1, "reflection_memory_index": -1} if run_mode==RunMode.Train else                {"investment_decision": "hold", "summary_reason": f"(error: {e})", "short_memory_index": -1,
                "mid_memory_index": -1, "long_memory_index": -1, "reflection_memory_index": -1}
    parsed = _extract_json(raw)
    try:
        if run_mode == RunMode.Train:
            return TrainReflectionOut.model_validate(parsed).model_dump()
        else:
            return TestReflectionOut.model_validate(parsed).model_dump()
    except ValidationError as ve:
        logger = logger or logging.getLogger(__name__)
        logger.warning(f"Validation error in reflection output: {ve}\nRaw: {raw}")
        if run_mode == RunMode.Train:
            return {"summary_reason": parsed.get("summary_reason",""),
                    "short_memory_index": int(parsed.get("short_memory_index",-1) or -1),
                    "mid_memory_index": int(parsed.get("mid_memory_index",-1) or -1),
                    "long_memory_index": int(parsed.get("long_memory_index",-1) or -1),
                    "reflection_memory_index": int(parsed.get("reflection_memory_index",-1) or -1)}
        else:
            decision = parsed.get("investment_decision","hold")
            if decision not in ("buy","sell","hold"): decision = "hold"
            return {"investment_decision": decision,
                    "summary_reason": parsed.get("summary_reason",""),
                    "short_memory_index": int(parsed.get("short_memory_index",-1) or -1),
                    "mid_memory_index": int(parsed.get("mid_memory_index",-1) or -1),
                    "long_memory_index": int(parsed.get("long_memory_index",-1) or -1),
                    "reflection_memory_index": int(parsed.get("reflection_memory_index",-1) or -1)}
