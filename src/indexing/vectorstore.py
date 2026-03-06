import logging
import os
import uuid
from typing import Any, Dict, List, Optional

from .embedder import Embedder

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        FieldCondition,
        Filter,
        MatchValue,
        PointStruct,
        VectorParams,
    )

    HAS_QDRANT = True
except Exception:  # pragma: no cover - optional dependency
    HAS_QDRANT = False
    QdrantClient = None
    Distance = FieldCondition = Filter = MatchValue = PointStruct = VectorParams = None


logger = logging.getLogger(__name__)


class VectorStore:
    """Manage vector storage and retrieval with Qdrant or in-memory fallback."""

    def __init__(self):
        self.embedder = Embedder()
        self.collection_name = os.getenv("QDRANT_COLLECTION", "multimodal_rag")
        self._use_qdrant = False
        self.client = None
        self._memory_points: List[Dict[str, Any]] = []

        if HAS_QDRANT:
            try:
                url = os.getenv("QDRANT_URL", ":memory:")
                self.client = QdrantClient(url)
                self._use_qdrant = True
                self._init_collection()
            except Exception as e:
                logger.warning(
                    "Qdrant unavailable (%s). Falling back to in-memory vector store.", e
                )
                self._use_qdrant = False
        else:
            logger.warning(
                "qdrant_client is not installed. Using in-memory vector store fallback."
            )

    def _init_collection(self):
        """Initialize Qdrant collection."""
        try:
            collections = self.client.get_collections().collections
            if not any(c.name == self.collection_name for c in collections):
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
                )
                logger.info(f"Created collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to initialize collection: {e}")
            raise

    def index_elements(
        self,
        elements: List[Dict[str, Any]],
        doc_name: str = None,
        doc_id: Optional[str] = None,
    ) -> str:
        """Index document elements with embeddings."""
        doc_id = doc_id or str(uuid.uuid4())
        qdrant_points = []

        for element in elements:
            text = self._get_text_content(element)
            if not text:
                continue

            payload = self._build_payload(element, text, doc_id=doc_id, doc_name=doc_name)
            embedding = self._safe_embed(text)

            # Always store in memory fallback for resilience.
            self._memory_points.append(
                {"id": str(uuid.uuid4()), "vector": embedding, "payload": payload}
            )

            if self._use_qdrant and embedding:
                qdrant_points.append(
                    PointStruct(id=str(uuid.uuid4()), vector=embedding, payload=payload)
                )

        if self._use_qdrant and qdrant_points:
            self.client.upsert(collection_name=self.collection_name, points=qdrant_points)
            logger.info(f"Indexed {len(qdrant_points)} elements for doc {doc_id}")
        elif not self._use_qdrant:
            logger.info(
                f"Indexed {sum(1 for p in self._memory_points if p['payload'].get('doc_id') == doc_id)} "
                f"elements for doc {doc_id} in memory store"
            )

        return doc_id

    def search(self, query: str, limit: int = 5, doc_id: str = None) -> List[Dict[str, Any]]:
        """Search for relevant elements."""
        query_vector = self._safe_embed(query)

        if self._use_qdrant and query_vector:
            try:
                query_filter = None
                if doc_id:
                    query_filter = Filter(
                        must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
                    )
                results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=limit,
                    query_filter=query_filter,
                )
                return [hit.payload for hit in results]
            except Exception as e:
                logger.error(f"Qdrant search failed; falling back to memory search: {e}")

        return self._search_memory(query, query_vector, limit, doc_id=doc_id)

    def get_document_stats(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Return metadata and indexing stats for a document."""
        if self._use_qdrant:
            try:
                doc_filter = Filter(
                    must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
                )
                count = self.client.count(
                    collection_name=self.collection_name, count_filter=doc_filter
                ).count
                if count == 0:
                    return None

                pages = set()
                doc_name = None
                next_offset = None
                while True:
                    points, next_offset = self.client.scroll(
                        collection_name=self.collection_name,
                        scroll_filter=doc_filter,
                        with_payload=True,
                        with_vectors=False,
                        limit=256,
                        offset=next_offset,
                    )
                    for point in points:
                        payload = point.payload or {}
                        if doc_name is None and payload.get("doc_name"):
                            doc_name = payload["doc_name"]
                        if payload.get("page") is not None:
                            pages.add(payload["page"])
                    if next_offset is None:
                        break

                return {
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "elements_indexed": int(count),
                    "pages": sorted(pages),
                }
            except Exception as e:
                logger.error(f"Failed to fetch document stats for {doc_id}: {e}")

        doc_points = [
            point for point in self._memory_points if point["payload"].get("doc_id") == doc_id
        ]
        if not doc_points:
            return None
        doc_name = next(
            (p["payload"].get("doc_name") for p in doc_points if p["payload"].get("doc_name")),
            None,
        )
        pages = sorted(
            {
                p["payload"].get("page")
                for p in doc_points
                if p["payload"].get("page") is not None
            }
        )
        return {
            "doc_id": doc_id,
            "doc_name": doc_name,
            "elements_indexed": len(doc_points),
            "pages": pages,
        }

    def list_documents(self) -> List[Dict[str, Any]]:
        """List indexed documents with element counts and page coverage."""
        if self._use_qdrant:
            docs: Dict[str, Dict[str, Any]] = {}
            try:
                next_offset = None
                while True:
                    points, next_offset = self.client.scroll(
                        collection_name=self.collection_name,
                        with_payload=True,
                        with_vectors=False,
                        limit=256,
                        offset=next_offset,
                    )

                    for point in points:
                        payload = point.payload or {}
                        doc_id = payload.get("doc_id")
                        if not doc_id:
                            continue

                        if doc_id not in docs:
                            docs[doc_id] = {
                                "doc_id": doc_id,
                                "doc_name": payload.get("doc_name"),
                                "elements_indexed": 0,
                                "pages": set(),
                            }

                        doc = docs[doc_id]
                        doc["elements_indexed"] += 1
                        if not doc.get("doc_name") and payload.get("doc_name"):
                            doc["doc_name"] = payload.get("doc_name")
                        if payload.get("page") is not None:
                            doc["pages"].add(payload["page"])

                    if next_offset is None:
                        break
            except Exception as e:
                logger.error(f"Failed to list documents: {e}")
                return []

            output = []
            for doc in docs.values():
                output.append(
                    {
                        "doc_id": doc["doc_id"],
                        "doc_name": doc["doc_name"],
                        "elements_indexed": doc["elements_indexed"],
                        "pages": sorted(doc["pages"]),
                    }
                )
            output.sort(key=lambda x: (x.get("doc_name") or "", x["doc_id"]))
            return output

        docs: Dict[str, Dict[str, Any]] = {}
        for point in self._memory_points:
            payload = point["payload"]
            doc_id = payload.get("doc_id")
            if not doc_id:
                continue
            if doc_id not in docs:
                docs[doc_id] = {
                    "doc_id": doc_id,
                    "doc_name": payload.get("doc_name"),
                    "elements_indexed": 0,
                    "pages": set(),
                }
            docs[doc_id]["elements_indexed"] += 1
            if not docs[doc_id]["doc_name"] and payload.get("doc_name"):
                docs[doc_id]["doc_name"] = payload.get("doc_name")
            if payload.get("page") is not None:
                docs[doc_id]["pages"].add(payload["page"])

        output = []
        for doc in docs.values():
            output.append(
                {
                    "doc_id": doc["doc_id"],
                    "doc_name": doc["doc_name"],
                    "elements_indexed": doc["elements_indexed"],
                    "pages": sorted(doc["pages"]),
                }
            )
        output.sort(key=lambda x: (x.get("doc_name") or "", x["doc_id"]))
        return output

    def delete_document(self, doc_id: str) -> bool:
        """Delete all indexed points for a document."""
        success = True
        if self._use_qdrant:
            try:
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=Filter(
                        must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
                    ),
                )
            except Exception as e:
                logger.error(f"Failed to delete document {doc_id} from Qdrant: {e}")
                success = False

        before = len(self._memory_points)
        self._memory_points = [
            p for p in self._memory_points if p["payload"].get("doc_id") != doc_id
        ]
        if before == len(self._memory_points) and not self._use_qdrant:
            return False
        return success

    def _search_memory(
        self,
        query: str,
        query_vector: Optional[List[float]],
        limit: int,
        doc_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        candidates = []
        for point in self._memory_points:
            payload = point["payload"]
            if doc_id and payload.get("doc_id") != doc_id:
                continue
            score = self._score_point(query, query_vector, point)
            candidates.append((score, payload))
        candidates.sort(key=lambda item: item[0], reverse=True)
        return [payload for _, payload in candidates[:limit]]

    def _score_point(
        self,
        query: str,
        query_vector: Optional[List[float]],
        point: Dict[str, Any],
    ) -> float:
        vector = point.get("vector")
        if query_vector and vector:
            return sum(a * b for a, b in zip(query_vector, vector))
        content = (point["payload"].get("content") or "").lower()
        query_terms = [term for term in query.lower().split() if term]
        if not query_terms:
            return 0.0
        hits = sum(1 for term in query_terms if term in content)
        return hits / len(query_terms)

    def _safe_embed(self, text: str) -> Optional[List[float]]:
        try:
            return self.embedder.embed(text)
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return None

    def _build_payload(
        self, element: Dict[str, Any], text: str, doc_id: str, doc_name: Optional[str]
    ) -> Dict[str, Any]:
        return {
            "type": element.get("type"),
            "content": text[:1000],  # Truncate for storage
            "page": element.get("page"),
            "element_id": element.get("element_id"),
            "doc_id": doc_id,
            "doc_name": doc_name,
            "bbox": element.get("bbox"),
        }

    def _get_text_content(self, element: Dict[str, Any]) -> str:
        """Extract text content from element for embedding."""
        element_type = element.get("type")
        if element_type == "text":
            return element.get("content", "")
        if element_type in {"image", "chart"}:
            return element.get("description", "")
        if element_type == "table":
            content = element.get("content", {})
            if isinstance(content, dict):
                return content.get("summary", "")
            if isinstance(content, str):
                return content
        return ""
