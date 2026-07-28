"""
vector_store.py
----------------
Thin abstraction over the two supported vector backends:

  * Pinecone   -> used automatically if PINECONE_API_KEY is set in .env
  * FAISS      -> local, on-disk fallback used automatically if it is NOT set

Both ingest.py (writes) and rag_graph.py (reads) import this module so
there is exactly one place that knows how to talk to "the vector DB",
regardless of which one is actually active. This is what makes the
Pinecone-or-FAISS choice invisible to the rest of the app.

Embeddings: sentence-transformers/all-MiniLM-L6-v2 (384-dim, local, free,
no API key). Loaded once and cached at module level.
"""

import os
import pickle
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384 

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "").strip()
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agentic-ai-ebook")
USE_PINECONE = bool(PINECONE_API_KEY)

FAISS_DIR = Path("faiss_index")
FAISS_INDEX_PATH = FAISS_DIR / "index.faiss"
FAISS_META_PATH = FAISS_DIR / "metadata.pkl"


_model: SentenceTransformer | None = None


def get_embedder() -> SentenceTransformer:
    """Lazily load and cache the local embedding model."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def embed_texts(texts: List[str]) -> np.ndarray:
    """Embed a list of strings -> (n, EMBEDDING_DIM) float32 array, L2-normalized
    so that a plain dot product equals cosine similarity."""
    model = get_embedder()
    vectors = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return np.asarray(vectors, dtype="float32")



def _get_pinecone_index():
    from pinecone import Pinecone, ServerlessSpec

    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing = [idx["name"] for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),  # free-tier region
        )
    return pc.Index(PINECONE_INDEX_NAME)


def pinecone_count() -> int:
    """Number of vectors currently stored — used by ingest.py for idempotency."""
    index = _get_pinecone_index()
    stats = index.describe_index_stats()
    return int(stats.get("total_vector_count", 0))


def pinecone_upsert(chunks: List[Dict[str, Any]], vectors: np.ndarray) -> None:
    index = _get_pinecone_index()
    payload = []
    for chunk, vec in zip(chunks, vectors):
        payload.append(
            {
                "id": chunk["chunk_id"],
                "values": vec.tolist(),
                "metadata": {
                    "text": chunk["text"],
                    "page": chunk["page"],
                },
            }
        )
    
    for i in range(0, len(payload), 100):
        index.upsert(vectors=payload[i : i + 100])


def pinecone_query(query_vector: np.ndarray, k: int) -> List[Dict[str, Any]]:
    index = _get_pinecone_index()
    result = index.query(vector=query_vector.tolist(), top_k=k, include_metadata=True)
    hits = []
    for match in result["matches"]:
        hits.append(
            {
                "text": match["metadata"]["text"],
                "page": match["metadata"]["page"],
                "chunk_id": match["id"],
                "score": float(match["score"]), 
            }
        )
    return hits


def _load_faiss():
    import faiss

    if not FAISS_INDEX_PATH.exists():
        return None, []
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    with open(FAISS_META_PATH, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata


def faiss_count() -> int:
    index, _ = _load_faiss()
    return 0 if index is None else index.ntotal


def faiss_upsert(chunks: List[Dict[str, Any]], vectors: np.ndarray) -> None:
    import faiss

    FAISS_DIR.mkdir(exist_ok=True)
    index, metadata = _load_faiss()
    if index is None:
        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        metadata = []

    index.add(vectors)
    metadata.extend(chunks)

    faiss.write_index(index, str(FAISS_INDEX_PATH))
    with open(FAISS_META_PATH, "wb") as f:
        pickle.dump(metadata, f)


def faiss_query(query_vector: np.ndarray, k: int) -> List[Dict[str, Any]]:
    index, metadata = _load_faiss()
    if index is None or index.ntotal == 0:
        return []

    scores, indices = index.search(query_vector.reshape(1, -1), k)
    hits = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = metadata[idx]
        hits.append(
            {
                "text": chunk["text"],
                "page": chunk["page"],
                "chunk_id": chunk["chunk_id"],
                "score": float(score),
            }
        )
    return hits


def active_backend() -> str:
    return "pinecone" if USE_PINECONE else "faiss"


def existing_vector_count() -> int:
    """Used by ingest.py to decide whether ingestion has already run."""
    return pinecone_count() if USE_PINECONE else faiss_count()


def upsert_chunks(chunks: List[Dict[str, Any]], vectors: np.ndarray) -> None:
    if USE_PINECONE:
        pinecone_upsert(chunks, vectors)
    else:
        faiss_upsert(chunks, vectors)


def similarity_search(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """Embed the query and return the top-k most similar chunks, each with
    text, page, chunk_id, and a cosine-similarity score in ~[0, 1]."""
    query_vector = embed_texts([query])[0]
    if USE_PINECONE:
        return pinecone_query(query_vector, k)
    return faiss_query(query_vector, k)
