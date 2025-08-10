import os
from typing import Callable, Dict, Any, Union
from openai import OpenAI

class LongerThanContextError(Exception):
    pass

class ChatOpenAICompatible:
    def __init__(self, end_point: str, model: str, system_message: str="You are a helpful assistant.", other_parameters: Union[Dict[str, Any], None]=None):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model = model
        self.system_message = system_message
        self.other_parameters = {} if other_parameters is None else other_parameters

    def guardrail_endpoint(self) -> Callable:
        def end_point(input: str, **kwargs) -> str:
            messages = [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": input},
            ]
            temperature = float(self.other_parameters.get("temperature", 0.2))
            max_tokens = int(self.other_parameters.get("max_tokens", 1024))
            try:
                resp = self.client.responses.create(
                    model=self.model,
                    input={"messages": messages},
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )
            except Exception as e:
                msg = str(e).lower()
                if "token" in msg and ("max" in msg or "context" in msg):
                    raise LongerThanContextError
                raise
            if hasattr(resp, "output_text"):
                return resp.output_text
            out = []
            for item in getattr(resp, "output", []):
                if getattr(item, "type", "") == "message":
                    for c in item.message.content:
                        if getattr(c, "type", "") == "output_text":
                            out.append(c.text)
            return "".join(out) if out else ""
        return end_point
