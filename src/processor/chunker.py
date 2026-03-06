"""Text chunking utilities."""
from typing import List, Dict, Any

class TextChunker:
    """Chunk text into smaller pieces with overlap."""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks."""
        if not text:
            return []
        
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            
            chunk_data = {
                "content": chunk_text,
                "chunk_index": len(chunks),
                "start_word": i,
                "end_word": i + len(chunk_words)
            }
            
            if metadata:
                chunk_data.update(metadata)
            
            chunks.append(chunk_data)
        
        return chunks
