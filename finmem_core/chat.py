import os, json
from typing import Dict, Any, Optional, List
from tenacity import retry, wait_random_exponential, stop_after_attempt

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

def _client():
    if OpenAI is None:
        return None
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY", None))

@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(3))
def call_llm(messages: List[Dict[str,str]], model: str, json_schema: Optional[Dict]=None, max_output_tokens: int=500) -> str:
    """
    Tries Responses API first; falls back to Chat Completions with JSON best-effort.
    Returns the raw text content from the model.
    """
    client = _client()
    if client is None:
        # No OpenAI client, return a naive policy: hold
        return '{"action": "hold", "confidence": 0.0, "rationale": "No LLM key; defaulting to hold."}'

    # Shape messages for either API
    # Attempt Responses API
    try:
        if json_schema:
            resp = client.responses.create(
                model=model,
                input=[{"role": m["role"], "content": m["content"]} for m in messages],
                response_format={"type":"json_schema","json_schema":{"name":"trading_decision","schema":json_schema}},
                max_output_tokens=max_output_tokens,
            )
        else:
            resp = client.responses.create(
                model=model,
                input=[{"role": m["role"], "content": m["content"]} for m in messages],
                max_output_tokens=max_output_tokens,
            )
        # Extract text
        if resp and resp.output and len(resp.output) > 0:
            parts = []
            for o in resp.output:
                if getattr(o, "type", None) == "output_text":
                    parts.append(o.text)
            if parts:
                return "\n".join(parts)
        # fallback extraction
        return getattr(resp, "output_text", "") or ""
    except Exception:
        # Fallback to Chat Completions
        try:
            if json_schema:
                # force JSON via system instruction
                sys_msg = {"role":"system","content":"Return ONLY valid JSON for the following schema. Do not include any extra text."}
                mm = [sys_msg] + messages
            else:
                mm = messages
            cc = client.chat.completions.create(
                model=model,
                messages=mm,
                temperature=0.2
            )
            return cc.choices[0].message.content
        except Exception:
            return '{"action": "hold", "confidence": 0.0, "rationale": "LLM call failed; default hold."}'
