import json
from typing import List, Dict, Any
from .prompts import PERSONAS, build_user_prompt, TRADING_JSON_SCHEMA
from .chat import call_llm

class LLMAgent:
    def __init__(self, persona: str, model: str):
        self.persona = persona
        self.system_prompt = PERSONAS[persona]["system"]
        self.model = model

    def step(self, symbol: str, date: str, price_window, news_list, retrieved_memories) -> Dict[str, Any]:
        messages = [
            {"role":"system","content": self.system_prompt},
            {"role":"user","content": build_user_prompt(symbol, date, price_window, news_list, retrieved_memories)}
        ]
        raw = call_llm(messages, model=self.model, json_schema=TRADING_JSON_SCHEMA)
        # Parse JSON best-effort
        try:
            # if model returns extra text, find first {...}
            s = raw
            l = s.find("{")
            r = s.rfind("}")
            if l >= 0 and r > l:
                s = s[l:r+1]
            data = json.loads(s)
        except Exception:
            data = {"action":"hold", "confidence":0.0, "rationale":"Failed to parse model JSON."}
        action = str(data.get("action","hold")).lower().strip()
        if action not in {"buy","sell","hold"}:
            action = "hold"
        conf = float(data.get("confidence",0.0))
        rationale = str(data.get("rationale","")).strip()
        mem_ids = data.get("memory_ids", [])
        return {"action": action, "confidence": conf, "rationale": rationale, "memory_ids": mem_ids, "raw": raw}
