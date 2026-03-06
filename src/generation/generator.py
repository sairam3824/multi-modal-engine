from typing import List, Dict, Any
import os

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional in lightweight environments
    OpenAI = None

class Generator:
    """Generate responses using retrieved multimodal context."""
    
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key) if (OpenAI and api_key) else None
        self.model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o")
    
    def generate(self, query: str, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate answer with multimodal context."""
        # Build prompt with context
        prompt = self._build_prompt(query, context)
        
        # Generate response
        if self.client:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided document context. Always cite sources with page numbers."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            answer = response.choices[0].message.content
        else:
            answer = self._fallback_answer(query, context)
        
        return {
            "answer": answer,
            "sources": self._extract_sources(context)
        }

    def _fallback_answer(self, query: str, context: List[Dict[str, Any]]) -> str:
        """Best-effort local answer when OpenAI client is unavailable."""
        if not context:
            return "No relevant information found."
        top = context[0]
        page = top.get("page", "?")
        content = (top.get("content") or "").strip()
        snippet = content[:350] if content else "No textual content available."
        return (
            f"OpenAI chat client is not configured, so this is a local summary for '{query}'. "
            f"Top source (page {page}): {snippet}"
        )
    
    def _build_prompt(self, query: str, context: List[Dict[str, Any]]) -> str:
        """Build prompt with multimodal context."""
        prompt = f"Question: {query}\n\nContext:\n"
        
        for idx, item in enumerate(context, 1):
            element_type = item.get("type", "unknown")
            page = item.get("page", "?")
            content = item.get("content", "")
            
            prompt += f"\n[Source {idx} - {element_type.upper()} from page {page}]\n"
            prompt += f"{content[:500]}\n"
        
        prompt += "\nAnswer the question based on the context above. Cite sources like 'According to the chart on page 3...'."
        return prompt
    
    def _extract_sources(self, context: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract source metadata."""
        return [{
            "type": item.get("type"),
            "page": item.get("page"),
            "element_id": item.get("element_id")
        } for item in context]
