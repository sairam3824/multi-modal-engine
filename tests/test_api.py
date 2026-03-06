"""Tests for API endpoints."""
import pytest
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_root():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data

def test_query_without_documents():
    """Test query when no documents are indexed."""
    response = client.post(
        "/query",
        json={"query": "test query", "top_k": 5}
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data


def test_upload_rejects_non_pdf():
    """Upload endpoint should reject non-PDF files."""
    response = client.post(
        "/upload",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 400
    assert "Only PDF files are supported" in response.text


def test_status_unknown_document():
    """Status endpoint returns 404 for unknown docs."""
    response = client.get("/status/does-not-exist")
    assert response.status_code == 404


def test_list_documents_endpoint_shape():
    """Documents endpoint should always return a stable response shape."""
    response = client.get("/documents")
    assert response.status_code == 200
    data = response.json()
    assert "count" in data
    assert "documents" in data
