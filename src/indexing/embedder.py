from typing import List
import os
import hashlib

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - dependency may be absent in lightweight envs
    OpenAI = None

class Embedder:
    """Generate embeddings for text content."""
    
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key) if (OpenAI and api_key) else None
        self.model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self.fallback_dim = 1536
    
    def embed(self, text: str) -> List[float]:
        """Generate embedding for single text."""
        if self.client:
            response = self.client.embeddings.create(model=self.model, input=text)
            return response.data[0].embedding
        return self._fallback_embed(text)
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if self.client:
            response = self.client.embeddings.create(model=self.model, input=texts)
            return [item.embedding for item in response.data]
        return [self._fallback_embed(text) for text in texts]

    def _fallback_embed(self, text: str) -> List[float]:
        """Generate deterministic local embeddings when OpenAI is unavailable."""
        text = text or ""
        vector = [0.0] * self.fallback_dim
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
            idx = int(digest[:8], 16) % self.fallback_dim
            vector[idx] += 1.0
        norm = sum(v * v for v in vector) ** 0.5
        if norm > 0:
            vector = [v / norm for v in vector]
        return vector
