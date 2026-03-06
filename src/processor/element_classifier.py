from typing import Dict, Any, List

class ElementClassifier:
    """Classify document elements into text, image, table, or chart."""
    
    def classify(self, element: Dict[str, Any]) -> str:
        """Determine element type with enhanced classification."""
        element_type = element.get("type", "text")
        
        if element_type in ["text", "table", "chart"]:
            return element_type
        
        if element_type == "image":
            return self._classify_image(element)
        
        return "text"
    
    def _classify_image(self, element: Dict[str, Any]) -> str:
        """Distinguish between regular images and charts."""
        textual_hints: List[str] = []
        for key in ("caption", "title", "context", "description", "content"):
            value = element.get(key)
            if isinstance(value, str):
                textual_hints.append(value.lower())

        hint_text = " ".join(textual_hints)
        chart_keywords = {"chart", "graph", "plot", "axis", "trend", "histogram", "bar", "line"}
        if any(keyword in hint_text for keyword in chart_keywords):
            return "chart"

        return "image"
