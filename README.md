# Multimodal RAG Engine

A production-ready RAG pipeline that processes PDFs with text, images, tables, and charts. Uses GPT-4o Vision to understand visual content, generates text descriptions for indexing, and provides unified retrieval across all modalities.

> Built on multimodal research presented at ICISML 2026

## Features

- **Multimodal Document Processing** - Extracts and classifies text blocks, images, tables, and charts from PDFs
- **Vision-Powered Understanding** - Uses GPT-4o Vision to generate semantic descriptions of images and charts
- **Intelligent Table Extraction** - Converts tables to structured data with text summaries
- **Hybrid Retrieval** - Searches across all modalities with query routing and re-ranking
- **Document-Scoped Querying** - Query globally or constrain retrieval using `doc_id`
- **Document Lifecycle APIs** - Upload, list, status, and delete document indexes
- **Robust Runtime Fallbacks** - Runs without OpenAI/Qdrant using local in-memory retrieval mode
- **Rich Context Assembly** - Interleaves text and visual descriptions for comprehensive answers
- **Source Attribution** - Provides precise citations with page numbers and element types
- **REST API** - FastAPI server with comprehensive endpoints
- **Interactive Demo** - Streamlit UI for document upload and chat interface
- **Docker Ready** - Complete containerization with docker-compose

## Quick Start

Prefer a minimal command list? See [`torun.txt`](torun.txt).

### Prerequisites

- Python 3.11 recommended (works best with pinned dependencies)
- OpenAI API key with GPT-4o access (optional for local fallback mode)

### Installation

```bash
# Clone repository
git clone <your-repo-url>
cd multi-modal-engine

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Runtime Modes

- **Full mode (recommended)**: OpenAI + Qdrant available
  - Best response quality and semantic retrieval
- **Fallback mode**: missing OpenAI and/or Qdrant
  - Uses deterministic local embeddings, in-memory vector search, and local text summary fallback
  - Good for local development and API validation without external services

### Running the System

**Option 1: API Server**
```bash
uvicorn src.api:app --reload
# API available at http://localhost:8000
```

**Option 2: Streamlit Demo**
```bash
streamlit run src/demo.py
# Demo available at http://localhost:8501
```

**Option 3: Docker**
```bash
export OPENAI_API_KEY=your-key-here
docker-compose up -d
# API: http://localhost:8000
# Demo: http://localhost:8501
```

## Usage

### API Endpoints

**Upload Document**
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@document.pdf"
```

**Query Documents**
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What does the revenue chart show?", "top_k": 5}'
```

**Query a Specific Document**
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query":"Summarize section 2","top_k":5,"doc_id":"<doc-id-from-upload>"}'
```

**Health Check**
```bash
curl http://localhost:8000/health
```

**Delete Document**
```bash
curl -X DELETE "http://localhost:8000/documents/{doc_id}"
```

**List Documents**
```bash
curl http://localhost:8000/documents
```

**Document Status**
```bash
curl http://localhost:8000/status/{doc_id}
```

**Batch Upload**
```bash
curl -X POST "http://localhost:8000/batch-upload" \
  -F "files=@doc1.pdf" \
  -F "files=@doc2.pdf"
```

### Python SDK

```python
from src.api_client import MultimodalRAGClient

# Initialize client
client = MultimodalRAGClient("http://localhost:8000")

# Check health
health = client.health_check()
print(health)

# Upload document
result = client.upload_document("document.pdf")
print(f"Processed {result['elements_processed']} elements")

# Query
response = client.query("What are the main findings?")
print(response["answer"])
print(f"Sources: {len(response['sources'])}")

# Query only one uploaded document
response = client.query("Summarize section 2", doc_id=result["doc_id"])

# List indexed documents
print(client.list_documents())

# Document status
print(client.get_status(result["doc_id"]))

# Delete document
print(client.delete_document(result["doc_id"]))
```

### Direct Usage

```python
from src.processor.pdf_parser import PDFParser
from src.processor.image_describer import ImageDescriber
from src.indexing.vectorstore import VectorStore
from src.retrieval.searcher import Searcher
from src.generation.generator import Generator

# Process document
parser = PDFParser()
elements = parser.parse("document.pdf")

# Describe images
describer = ImageDescriber()
for element in elements:
    if element["type"] == "image":
        element["description"] = describer.describe(element["content"])

# Index
store = VectorStore()
doc_id = store.index_elements(elements, doc_name="document.pdf")

# Search and generate
searcher = Searcher(store)
generator = Generator()

results = searcher.search("What are the key findings?")
response = generator.generate("What are the key findings?", results)

print(response["answer"])
```

## Architecture

```
Document → Parser → Classifier → Processors → Embeddings → Vector Store
                                 (Text/Image/Table/Chart)

Query → Router → Retrieval → Re-ranking → Generation → Response
```

### Components

- **Document Processor** (`src/processor/`)
  - PDF parsing with PyMuPDF
  - Element classification
  - GPT-4o Vision for image descriptions
  - Table extraction and summarization

- **Indexing** (`src/indexing/`)
  - OpenAI text embeddings (text-embedding-3-small)
  - Qdrant vector storage
  - Deterministic local embedding + in-memory vector fallback when dependencies are unavailable
  - Metadata management

- **Retrieval** (`src/retrieval/`)
  - Query routing (visual/data/text)
  - Multimodal search
  - Result re-ranking

- **Generation** (`src/generation/`)
  - GPT-4o response generation
  - Source citations

- **API & Demo**
  - FastAPI REST API
  - Streamlit web interface

## Configuration

Set these environment variables in `.env`:

```bash
# Recommended for full mode
OPENAI_API_KEY=sk-your-key-here

# Optional
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-4o
QDRANT_URL=:memory:
QDRANT_API_KEY=
MAX_UPLOAD_SIZE_MB=50
LOG_LEVEL=INFO
```

Notes:
- If `OPENAI_API_KEY` is not set, the API still starts and serves fallback responses.
- If `qdrant-client` is unavailable or `QDRANT_URL` is unreachable, the API falls back to in-memory vector storage.

## Project Structure

```
multi-modal-engine/
├── src/
│   ├── processor/          # Document parsing and extraction
│   │   ├── pdf_parser.py
│   │   ├── element_classifier.py
│   │   ├── image_describer.py
│   │   ├── table_extractor.py
│   │   └── chunker.py
│   ├── indexing/           # Embeddings and storage
│   │   ├── embedder.py
│   │   └── vectorstore.py
│   ├── retrieval/          # Search and routing
│   │   ├── router.py
│   │   ├── searcher.py
│   │   └── reranker.py
│   ├── generation/         # Response generation
│   │   └── generator.py
│   ├── utils/              # Utilities
│   │   ├── config.py
│   │   └── logger.py
│   ├── api.py             # FastAPI server
│   ├── api_client.py      # Python SDK
│   └── demo.py            # Streamlit UI
├── tests/                 # Test suite
├── examples/              # Example usage
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_api.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

## Docker Deployment

### Using Docker Compose (Recommended)

```bash
# Set environment variable
export OPENAI_API_KEY=your-key-here

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

Services:
- API: http://localhost:8000
- Demo: http://localhost:8501
- Qdrant: http://localhost:6333

### Using Docker Only

```bash
# Build image
docker build -t multimodal-rag .

# Run API
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=your-key \
  multimodal-rag

# Run Demo
docker run -p 8501:8501 \
  -e OPENAI_API_KEY=your-key \
  multimodal-rag \
  streamlit run src/demo.py --server.port 8501 --server.address 0.0.0.0
```

## Production Deployment

### Recommendations

1. **Use Persistent Qdrant**
   - Don't use `:memory:` in production
   - Use Qdrant Cloud or self-hosted instance
   - Set `QDRANT_URL` and `QDRANT_API_KEY`

2. **Add Authentication**
   - Implement JWT or API key authentication
   - Use FastAPI security utilities
   - Protect sensitive endpoints

3. **Set Up Monitoring**
   - Use Prometheus for metrics
   - Set up logging aggregation
   - Monitor OpenAI API usage and costs

4. **Configure Rate Limiting**
   - Prevent abuse
   - Manage API costs
   - Use slowapi or similar

5. **Enable HTTPS**
   - Use reverse proxy (nginx, Caddy)
   - Configure SSL certificates
   - Enforce HTTPS only

6. **Optimize Performance**
   - Cache embeddings
   - Use async processing for uploads
   - Implement connection pooling

## Cost Considerations

OpenAI API costs:
- **Embeddings**: ~$0.02 per 1M tokens (text-embedding-3-small)
- **GPT-4o Vision**: ~$0.01 per image
- **GPT-4o Text**: ~$0.03 per 1K tokens

Tips to reduce costs:
- Cache image descriptions
- Batch embed operations
- Use smaller models where appropriate
- Monitor usage with OpenAI dashboard

## Troubleshooting

### Common Issues

**"OpenAI API key not found"**
- Ensure `.env` file exists
- Check `OPENAI_API_KEY` is set correctly
- Restart the server
- If you intentionally run without OpenAI, fallback mode will still work (lower answer quality)

**"Qdrant connection failed"**
- Verify Qdrant is running (if using docker-compose)
- Check `QDRANT_URL` configuration
- If unavailable, the app automatically uses in-memory fallback

**"PDF parsing failed"**
- Ensure PDF is not corrupted
- Check if PDF has extractable text
- Try a different PDF

**"ModuleNotFoundError: frontend" when importing `fitz`**
- Uninstall conflicting package: `pip uninstall -y fitz`
- Install PyMuPDF: `pip install pymupdf==1.23.26`

**PyMuPDF build failures on Python 3.13**
- Use Python 3.11 in a virtual environment for best compatibility

**"Demo shows API not running"**
- Start API server first: `uvicorn src.api:app --reload`
- Check API is accessible at http://localhost:8000
- Verify health endpoint: `curl http://localhost:8000/health`

## Development

### Setup Development Environment

```bash
# Install dev dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black flake8

# Format code
black src/ tests/

# Lint
flake8 src/ tests/

# Run tests
pytest tests/ -v
```

### Adding New Features

1. Create feature branch
2. Implement changes
3. Add tests
4. Update documentation
5. Submit pull request

## Performance

Expected performance:
- **Query response**: < 2 seconds
- **Small PDF (10 pages)**: ~30 seconds
- **Medium PDF (50 pages)**: ~2 minutes
- **Large PDF (100+ pages)**: ~5+ minutes

Factors affecting performance:
- Document size and complexity
- Number of images/tables
- OpenAI API response time
- Network latency

## Limitations

1. **Table Extraction**: Improved with PyMuPDF table detection, but still not perfect for all layouts
2. **Document Status Registry**: In-memory status cache is not persisted across server restarts
3. **Image Storage**: Only descriptions are indexed, not original image bytes
4. **Fallback Retrieval**: In-memory fallback is for development; use persistent Qdrant for production

## Future Enhancements

- Support for DOCX, HTML, and other formats
- Advanced table extraction with table-transformer
- Image-to-image search with CLIP embeddings
- Async processing with job queues
- Document registry database
- Caching layer for embeddings
- Advanced re-ranking models
- Multi-language support

## Tech Stack

- **Backend**: Python 3.11, FastAPI, Uvicorn
- **Frontend**: Streamlit
- **Document Processing**: PyMuPDF, unstructured.io, Pillow, pandas
- **Vector Store**: Qdrant
- **AI Models**: OpenAI (GPT-4o, text-embedding-3-small) with local fallbacks for offline/dev mode
- **Deployment**: Docker, docker-compose

## License

Apache License 2.0 - See LICENSE file for details.

## Citation

If you use this in research, please cite:

```
Multimodal RAG Engine (2026)
Based on research presented at ICISML 2026
```

## Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check troubleshooting section above
- Review example code in `examples/`

## Acknowledgments

Built on multimodal research presented at ICISML 2026, translating academic insights into a practical RAG system for real-world document understanding.
