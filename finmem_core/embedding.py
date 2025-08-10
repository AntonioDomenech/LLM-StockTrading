import os
from typing import List, Union
import numpy as np
from openai import OpenAI
class OpenAILongerThanContextEmb:
    def __init__(self, model_name: str = "text-embedding-3-small", chunk_char_size: int = 6000, **kwargs):
        self.model = model_name
        self.chunk_char_size = int(chunk_char_size)
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    def get_query_embedding(self, text: str) -> List[float]:
        return self._embed_one(text)
    def get_document_embedding(self, text: str) -> List[float]:
        return self._embed_one(text)
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_one(t) for t in texts]
    def __call__(self, text: Union[List[str], str]) -> np.ndarray:
        if isinstance(text, str): text = [text]
        vecs = [self._embed_one(t) for t in text]
        return np.array(vecs, dtype=np.float32)
    def get_embedding_dimension(self) -> int:
        if self.model in ("text-embedding-3-small","text-embedding-ada-002"): return 1536
        if self.model == "text-embedding-3-large": return 3072
        return 1536
    def _embed_one(self, text: str) -> List[float]:
        if not isinstance(text, str): text = str(text) if text is not None else ""
        chunks = self._chunk_text(text, self.chunk_char_size) or [""]
        resp = self.client.embeddings.create(model=self.model, input=chunks)
        mats = [np.array(item.embedding, dtype=np.float32) for item in resp.data]
        if len(mats)==1: return mats[0].tolist()
        return np.mean(mats, axis=0).astype(np.float32).tolist()
    @staticmethod
    def _chunk_text(text: str, size: int) -> List[str]:
        if size <= 0: return [text]
        n = len(text); return [text[i:i+size] for i in range(0, n, size)]
