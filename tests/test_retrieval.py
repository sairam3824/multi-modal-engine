import pytest
from src.retrieval.router import QueryRouter
from src.retrieval.reranker import Reranker

def test_query_router():
    router = QueryRouter()
    
    # Visual query
    result = router.route("What does the chart show?")
    assert result["needs_visual"] == True
    
    # Text query
    result = router.route("What is the main conclusion?")
    assert result["needs_text"] == True

    data_result = router.route("What trend does the data show?")
    assert data_result["needs_data"] == True

def test_reranker():
    reranker = Reranker()
    
    results = [
        {"type": "text", "page": 1, "content": "First"},
        {"type": "image", "page": 2, "content": "Second"},
        {"type": "text", "page": 1, "content": "Third"}
    ]
    
    reranked = reranker.rerank("test query", results)
    assert len(reranked) == 3
