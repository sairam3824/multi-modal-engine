import base64
from typing import Dict, Any
import os

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional in lightweight environments
    OpenAI = None

class ImageDescriber:
    """Generate semantic descriptions of images using GPT-4o Vision."""
    
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key) if (OpenAI and api_key) else None
        self.model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o")
    
    def describe(self, image_bytes: bytes, element_type: str = "image") -> str:
        """Generate description of image or chart."""
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        prompt = self._get_prompt(element_type)
        if not self.client:
            return "Image description unavailable: OpenAI client is not configured."
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }],
            max_tokens=300
        )
        
        return response.choices[0].message.content
    
    def _get_prompt(self, element_type: str) -> str:
        """Get appropriate prompt based on element type."""
        if element_type == "chart":
            return ("Describe this chart in detail. Include: type of chart, "
                   "axes labels, data trends, key insights, and specific values if visible.")
        else:
            return ("Describe this image in detail. Include: main subjects, "
                   "context, relationships, and any text or labels visible.")
