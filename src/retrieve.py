import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import chromadb
from sentence_transformers import SentenceTransformer
from config import TOP_K, SIMILARITY_THRESHOLD, CHROMA_PATH

_embedder = SentenceTransformer("all-MiniLM-L6-v2")

def get_collection():
    db = chromadb.PersistentClient(path=CHROMA_PATH)
    return db.get_collection("scaler_rag")

def retrieve(query: str, collection=None) -> list[dict]:
    if collection is None:
        collection = get_collection()

    query_embedding = _embedder.encode([query]).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    for i in range(len(results["documents"][0])):
        similarity = 1 - results["distances"][0][i]
        chunks.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "chunk_index": results["metadatas"][0][i]["chunk_index"],
            "similarity": round(similarity, 4)
        })
    return chunks
