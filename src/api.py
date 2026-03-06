from datetime import datetime, timezone
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .generation.generator import Generator
from .indexing.vectorstore import VectorStore
from .processor.element_classifier import ElementClassifier
from .processor.image_describer import ImageDescriber
from .processor.pdf_parser import PDFParser
from .processor.table_extractor import TableExtractor
from .retrieval.searcher import Searcher

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Multimodal RAG Engine", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
vectorstore = VectorStore()
searcher = Searcher(vectorstore)
generator = Generator()
processing_status: Dict[str, Dict[str, Any]] = {}


def _set_doc_status(doc_id: str, status: str, **extra: Any) -> None:
    """Update in-memory document processing status."""
    current = processing_status.get(doc_id, {})
    current.update(
        {
            "doc_id": doc_id,
            "status": status,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    current.update(extra)
    processing_status[doc_id] = current


class QueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=25)
    doc_id: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    sources: list


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a PDF document."""
    doc_id = str(uuid.uuid4())
    original_filename = Path(file.filename or "uploaded.pdf").name
    _set_doc_status(doc_id, "processing", filename=original_filename)

    try:
        if not original_filename.lower().endswith(".pdf"):
            raise HTTPException(400, "Only PDF files are supported")

        # Save uploaded file
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        stored_filename = f"{doc_id}_{original_filename}"
        file_path = upload_dir / stored_filename

        logger.info(f"Uploading file: {original_filename}")

        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Process document
        parser = PDFParser()
        classifier = ElementClassifier()
        describer = ImageDescriber()
        table_extractor = TableExtractor()

        logger.info("Parsing PDF...")
        elements = parser.parse(str(file_path))

        # Process each element
        logger.info(f"Processing {len(elements)} elements...")
        for element in elements:
            element_type = classifier.classify(element)
            element["type"] = element_type

            if element_type in {"image", "chart"} and element.get("content"):
                try:
                    logger.info(f"Describing image on page {element.get('page')}")
                    description = describer.describe(element["content"], element_type)
                    element["description"] = description
                    # Remove raw bytes after description
                    element["content"] = None
                except Exception as e:
                    logger.error(f"Failed to describe image: {e}")
                    element["description"] = "Image description unavailable"
                    element["content"] = None

            elif element_type == "table":
                try:
                    table_data = table_extractor.extract(element.get("content", {}))
                    element["content"] = table_data
                except Exception as e:
                    logger.error(f"Failed to extract table: {e}")

        # Index elements
        logger.info("Indexing elements...")
        doc_id = vectorstore.index_elements(
            elements, doc_name=original_filename, doc_id=doc_id
        )
        stats = vectorstore.get_document_stats(doc_id) or {}
        _set_doc_status(
            doc_id,
            "completed",
            filename=original_filename,
            elements_indexed=stats.get("elements_indexed", 0),
            pages=stats.get("pages", []),
        )

        logger.info(f"Successfully processed {original_filename}")

        return {
            "status": "success",
            "doc_id": doc_id,
            "filename": original_filename,
            "elements_processed": len(elements),
            "elements_indexed": stats.get("elements_indexed", 0),
        }

    except HTTPException as e:
        _set_doc_status(doc_id, "failed", filename=original_filename, error=e.detail)
        raise
    except Exception as e:
        _set_doc_status(doc_id, "failed", filename=original_filename, error=str(e))
        logger.error(f"Error processing document: {e}")
        raise HTTPException(500, f"Failed to process document: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query indexed documents."""
    try:
        logger.info(f"Query: {request.query}")

        if request.doc_id:
            doc_stats = vectorstore.get_document_stats(request.doc_id)
            if not doc_stats:
                raise HTTPException(404, f"Document not found: {request.doc_id}")

        # Search
        results = searcher.search(
            request.query, top_k=request.top_k, doc_id=request.doc_id
        )

        if not results:
            return QueryResponse(
                answer="No relevant information found. Please upload a document first.",
                sources=[],
            )

        # Generate answer
        response = generator.generate(request.query, results)
        return QueryResponse(**response)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(500, f"Failed to process query: {str(e)}")


@app.get("/status/{doc_id}")
async def get_status(doc_id: str):
    """Get processing status."""
    if doc_id in processing_status:
        return processing_status[doc_id]

    stats = vectorstore.get_document_stats(doc_id)
    if stats:
        status = {
            "doc_id": doc_id,
            "status": "completed",
            "filename": stats.get("doc_name"),
            "elements_indexed": stats.get("elements_indexed", 0),
            "pages": stats.get("pages", []),
        }
        processing_status[doc_id] = status
        return status

    raise HTTPException(404, f"Document not found: {doc_id}")


@app.get("/")
async def root():
    return {"message": "Multimodal RAG Engine API", "version": "1.0.0", "status": "running"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check if OpenAI API key is set
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return {"status": "unhealthy", "error": "OPENAI_API_KEY not set"}

        # Check Qdrant connection
        vectorstore_component = "in_memory"
        if getattr(vectorstore, "_use_qdrant", False):
            vectorstore.client.get_collections()
            vectorstore_component = "qdrant"

        return {
            "status": "healthy",
            "components": {
                "api": "ok",
                "vectorstore": vectorstore_component,
                "openai": "configured",
            },
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@app.get("/documents")
async def list_documents():
    """List all indexed documents."""
    try:
        docs = vectorstore.list_documents()
        return {"count": len(docs), "documents": docs}
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(500, str(e))


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document and its indexed elements."""
    try:
        deleted = vectorstore.delete_document(doc_id)
        if not deleted:
            raise HTTPException(500, "Failed to delete document")
        _set_doc_status(doc_id, "deleted")

        logger.info(f"Deleted document: {doc_id}")
        return {"status": "success", "doc_id": doc_id, "message": "Document deleted"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(500, str(e))


@app.post("/batch-upload")
async def batch_upload_documents(files: List[UploadFile] = File(...)):
    """Upload and process multiple PDF documents."""
    results = []

    for file in files:
        try:
            result = await upload_document(file)
            results.append(
                {"filename": file.filename, "status": "success", "result": result}
            )
        except HTTPException as e:
            logger.error(f"Failed to process {file.filename}: {e.detail}")
            results.append(
                {"filename": file.filename, "status": "error", "error": e.detail}
            )
        except Exception as e:
            logger.error(f"Failed to process {file.filename}: {e}")
            results.append(
                {"filename": file.filename, "status": "error", "error": str(e)}
            )

    return {
        "total": len(files),
        "successful": sum(1 for r in results if r["status"] == "success"),
        "failed": sum(1 for r in results if r["status"] == "error"),
        "results": results,
    }
