PERSONAS = {
    "Secure": {
        "system": (
            "You are a cautious equity trading assistant. Your priority is capital preservation, "
            "minimizing drawdowns, and avoiding overtrading. Prefer HOLD unless evidence is strong."
        )
    },
    "Balanced": {
        "system": (
            "You are a balanced equity trading assistant. You weigh risks and rewards fairly, "
            "seeking reasonable entries and exits. Avoid extreme risk."
        )
    },
    "Risk": {
        "system": (
            "You are an aggressive equity trading assistant. You are comfortable taking calculated risks "
            "when signals and news are favorable. Avoid reckless or random trades."
        )
    }
}

# JSON schema for trading decision
TRADING_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["buy","sell","hold"]
        },
        "confidence": {
            "type": "number"
        },
        "rationale": {
            "type": "string"
        },
        "memory_ids": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": ["action","confidence","rationale"],
    "additionalProperties": False
}

def build_user_prompt(symbol, date, price_window, news_list, retrieved_memories):
    # price_window: list[ (date_str, close) ]
    # news_list: list of dicts
    # retrieved_memories: list of {layer,text,score,meta}
    lines = []
    lines.append(f"Make a single-day trading decision for {symbol} on {date}.")
    lines.append("Recent prices (oldest→newest):")
    for d,p in price_window:
        lines.append(f"- {d}: {p:.2f}")
    if news_list:
        lines.append("\nRelevant news:")
        for i, n in enumerate(news_list, 1):
            t = n.get('title','').strip()
            s = n.get('source','')
            lines.append(f"- [{i}] {t} ({s})")
    if retrieved_memories:
        lines.append("\nRetrieved memories:")
        for i, m in enumerate(retrieved_memories, 1):
            layer = m.get("layer")
            score = m.get("score")
            txt = m.get("text","").strip().replace("\n"," ")
            lines.append(f"- [{i}] ({layer}, score={score:.3f}) {txt[:200]}")
    lines.append("\nRespond in JSON: {action: buy|sell|hold, confidence: 0..1, rationale: str, memory_ids: [ids if any]}")
    return "\n".join(lines)
