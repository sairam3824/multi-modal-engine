import requests
from typing import Dict, Any, Optional
from pathlib import Path

class MultimodalRAGClient:
    """Client for interacting with the Multimodal RAG API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health status."""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def upload_document(self, file_path: str) -> Dict[str, Any]:
        """Upload and process a PDF document."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(path, "rb") as f:
            files = {"file": (path.name, f, "application/pdf")}
            response = requests.post(
                f"{self.base_url}/upload",
                files=files,
                timeout=300
            )
            response.raise_for_status()
            return response.json()
    
    def query(self, query: str, top_k: int = 5, doc_id: Optional[str] = None) -> Dict[str, Any]:
        """Query indexed documents."""
        payload = {"query": query, "top_k": top_k}
        if doc_id:
            payload["doc_id"] = doc_id

        response = requests.post(
            f"{self.base_url}/query",
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    
    def get_status(self, doc_id: str) -> Dict[str, Any]:
        """Get document processing status."""
        response = requests.get(f"{self.base_url}/status/{doc_id}")
        response.raise_for_status()
        return response.json()

    def list_documents(self) -> Dict[str, Any]:
        """List indexed documents."""
        response = requests.get(f"{self.base_url}/documents")
        response.raise_for_status()
        return response.json()

    def delete_document(self, doc_id: str) -> Dict[str, Any]:
        """Delete indexed elements for a document."""
        response = requests.delete(f"{self.base_url}/documents/{doc_id}")
        response.raise_for_status()
        return response.json()
