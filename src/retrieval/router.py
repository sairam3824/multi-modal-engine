from typing import Dict, Any

class QueryRouter:
    """Route queries to appropriate retrieval strategies."""
    
    def route(self, query: str) -> Dict[str, Any]:
        """Determine query type and retrieval strategy."""
        query_lower = query.lower()
        
        # Detect visual queries
        visual_keywords = ["image", "chart", "graph", "figure", "diagram", "table", "show"]
        needs_visual = any(kw in query_lower for kw in visual_keywords)
        
        # Detect data queries
        data_keywords = ["data", "number", "value", "statistic", "trend"]
        needs_data = any(kw in query_lower for kw in data_keywords)
        
        return {
            "needs_visual": needs_visual,
            "needs_data": needs_data,
            "needs_text": True,  # Always include text
            "query": query
        }
