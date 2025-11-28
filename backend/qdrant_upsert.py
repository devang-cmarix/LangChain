# backend/qdrant_upsert.py
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from typing import List, Dict, Any
import uuid

def ensure_qdrant_collection(client: QdrantClient, collection_name: str, vector_size: int):
    # Create collection if missing. Use COSINE distance for semantic similarity.
    existing = client.get_collections()
    coll_names = [c.name for c in existing.collections]
    if collection_name not in coll_names:
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=rest.VectorParams(size=vector_size, distance=rest.Distance.COSINE)
        )

def upsert_chunks_to_qdrant(client: QdrantClient, collection_name: str, embeddings: List[List[float]], metadatas: List[Dict[str,Any]]):
    """
    embeddings: list of vectors (floats)
    metadatas: list of dicts corresponding to each vector
    We'll generate unique string ids for each vector.
    """
    assert len(embeddings) == len(metadatas)
    points = []
    for vec, md in zip(embeddings, metadatas):
        # generate deterministic id if you want deduplication: e.g., hash of pdf_name+page+start_line
        # Here we use uuid:
        vector_id = str(uuid.uuid4())
        points.append(rest.PointStruct(id=vector_id, vector=vec, payload=md))
    # Qdrant accepts upsert in batches; client.upsert will handle
    client.upsert(collection_name=collection_name, points=points)
