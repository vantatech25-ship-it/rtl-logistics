"""
pinecone_client.py
Pinecone vector memory client for RTL Logistics.
Stores hub GNN feature embeddings for fast spatial similarity search.
LangGraph uses this to answer: "Which hub is most similar to current conditions?"
"""

from pinecone import Pinecone, ServerlessSpec
from config import PINECONE_API_KEY, PINECONE_INDEX
from typing import List, Dict, Any


def get_index():
    """Connect to or create the Pinecone index."""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing = [i.name for i in pc.list_indexes()]
    if PINECONE_INDEX not in existing:
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=8,  # matches hub_feature vector length below
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return pc.Index(PINECONE_INDEX)


def hub_to_vector(hub: Dict[str, Any]) -> List[float]:
    """
    Converts a hub's GNN feature dict (from the Rust API) into an 8-dim
    embedding vector for Pinecone storage.

    Vector dimensions:
      [0] load_factor          — current hub load 0-1
      [1] congestion_score     — GNN congestion 0-1
      [2] connectivity_norm    — normalised degree (degree / 10)
      [3] latitude_norm        — latitude normalised to -1..1
      [4] longitude_norm       — longitude normalised to -1..1
      [5] is_active            — 1.0 if active else 0.0
      [6] reserved_0           — placeholder for future feature
      [7] reserved_1           — placeholder for future feature
    """
    lat = hub.get("latitude", 0.0)
    lon = hub.get("longitude", 0.0)
    return [
        float(hub.get("load_factor", 0.0)),
        float(hub.get("congestion_score", 0.0)),
        float(hub.get("connectivity", 0)) / 10.0,
        (lat + 90.0) / 180.0,    # normalize lat to 0..1
        (lon + 180.0) / 360.0,   # normalize lon to 0..1
        1.0 if hub.get("active", True) else 0.0,
        0.0,  # reserved
        0.0,  # reserved
    ]


def upsert_hub_embedding(hub: Dict[str, Any]) -> None:
    """Store or update a hub's feature embedding in Pinecone."""
    index = get_index()
    vector = hub_to_vector(hub)
    index.upsert(vectors=[{
        "id": hub["hub_id"],
        "values": vector,
        "metadata": {
            "hub_id": hub["hub_id"],
            "load_factor": hub.get("load_factor"),
            "congestion_score": hub.get("congestion_score"),
            "connectivity": hub.get("connectivity"),
        },
    }])
    print(f"[Pinecone] Upserted embedding for hub: {hub['hub_id']}")


def find_similar_hubs(reference_hub: Dict[str, Any], top_k: int = 3) -> List[Dict]:
    """
    Find hubs most similar to the reference hub's current state.
    Used by LangGraph to find alternative routing hubs when primary is congested.
    """
    index = get_index()
    vector = hub_to_vector(reference_hub)
    results = index.query(vector=vector, top_k=top_k, include_metadata=True)
    return results.get("matches", [])


def delete_hub_embedding(hub_id: str) -> None:
    """Remove a decommissioned hub from Pinecone."""
    index = get_index()
    index.delete(ids=[hub_id])
    print(f"[Pinecone] Deleted embedding for hub: {hub_id}")
