import os, json, math, hashlib
from typing import List, Dict, Tuple, Optional
import numpy as np

def _hash_embed(text: str, dim: int = 256) -> np.ndarray:
    """Deterministic local embedding (fallback when no OpenAI)."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    # repeat hash to fill dim
    rep = (dim + len(h) - 1) // len(h)
    vec = (h * rep)[:dim]
    arr = np.frombuffer(vec, dtype=np.uint8).astype(np.float32)
    arr = (arr - arr.mean()) / (arr.std() + 1e-6)
    return arr

class BrainDB:
    """
    Four 'layers': short, mid, long, reflections.
    Stores texts and embeddings; retrieve top-k by cosine similarity.
    """
    def __init__(self, embed_fn, dim: int = 256):
        self.embed_fn = embed_fn
        self.dim = dim
        self.store = {
            "short": [],
            "mid": [],
            "long": [],
            "reflections": []
        }  # each item: {"text": str, "vec": np.ndarray, "meta": dict}

    def add(self, layer: str, text: str, meta: Optional[dict] = None):
        if layer not in self.store:
            raise ValueError(f"Unknown layer {layer}")
        vec = self.embed_fn(text)
        self.store[layer].append({"text": text, "vec": vec, "meta": meta or {}})

    def retrieve(self, query: str, k: int = 8) -> List[Dict]:
        q = self.embed_fn(query)
        cands = []
        for layer, items in self.store.items():
            for it in items:
                sim = self._cosine(q, it["vec"])
                cands.append((sim, layer, it))
        cands.sort(key=lambda x: x[0], reverse=True)
        out = []
        for sim, layer, it in cands[:k]:
            d = {"layer": layer, "text": it["text"], "meta": it["meta"], "score": float(sim)}
            out.append(d)
        return out

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        na = np.linalg.norm(a) + 1e-9
        nb = np.linalg.norm(b) + 1e-9
        return float(np.dot(a, b) / (na * nb))

def make_embedder_openai_or_hash(openai_client=None, embed_model: str="text-embedding-3-small"):
    if openai_client is None:
        # hash fallback
        return lambda text: _hash_embed(text, 256)
    def _fn(text: str):
        try:
            # OpenAI Embeddings API
            out = openai_client.embeddings.create(model=embed_model, input=text)
            return np.array(out.data[0].embedding, dtype=np.float32)
        except Exception:
            # fallback
            return _hash_embed(text, 256)
    return _fn
