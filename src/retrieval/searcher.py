from typing import List, Dict, Any
from ..indexing.vectorstore import VectorStore
from .router import QueryRouter
from .reranker import Reranker

class Searcher:
    """Search across multimodal document elements."""
    
    def __init__(self, vectorstore: VectorStore):
        self.vectorstore = vectorstore
        self.router = QueryRouter()
        self.reranker = Reranker()
    
    def search(self, query: str, top_k: int = 5, doc_id: str = None) -> List[Dict[str, Any]]:
        """Execute multimodal search."""
        # Route query
        routing = self.router.route(query)
        
        # Retrieve candidates
        results = self.vectorstore.search(query, limit=top_k * 3, doc_id=doc_id)
        
        # Filter by modality if needed
        if routing["needs_visual"]:
            results = self._boost_visual(results)

        results = self.reranker.rerank(query, results)
        
        # Return top-k
        return results[:top_k]
    
    def _boost_visual(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Boost visual elements in results."""
        visual_types = {"image", "chart", "table"}
        visual = [r for r in results if r.get("type") in visual_types]
        text = [r for r in results if r.get("type") not in visual_types]
        return visual + text
