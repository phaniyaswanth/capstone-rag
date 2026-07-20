import sys

from pymongo import MongoClient
from openai import OpenAI
import certifi

from config import (
    MONGODB_URI,
    OPENROUTER_API_KEY,
    EMBEDDING_MODEL,
    SOURCE_DB,
    RAG_CHUNKS_COLLECTION,
)

mongo_client = MongoClient(MONGODB_URI, tlsCAFile=certifi.where())
db = mongo_client[SOURCE_DB]
chunks_col = db[RAG_CHUNKS_COLLECTION]

openai_client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)


def embed_query(text: str) -> list[float]:
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding


def vector_search(query: str, k: int = 5) -> list[dict]:
    query_embedding = embed_query(query)

    pipeline = [
        {
            "$vectorSearch": {
                "index": "rag_chunks_embedding",
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": k * 10,
                "limit": k,
            }
        },
        {
            "$project": {
                "text": 1,
                "listing_id": 1,
                "name": 1,
                "property_type": 1,
                "room_type": 1,
                "price": 1,
                "address": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]

    return list(chunks_col.aggregate(pipeline))


if __name__ == "__main__":
    test_queries = [
        "quiet apartment near the beach with wifi",
        "cheap private room in New York City",
        "family-friendly house with pool and kitchen",
    ]

    if len(sys.argv) > 1:
        test_queries = [" ".join(sys.argv[1:])]

    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 60)

        results = vector_search(query, k=5)

        if not results:
            print("No results returned. Check that the vector index is active.")
            continue

        for i, doc in enumerate(results, 1):
            addr = doc.get("address", {})
            print(
                f"  {i}. {doc.get('name', 'Unknown')} "
                f"| {doc.get('property_type', '')} / {doc.get('room_type', '')} "
                f"| ${doc.get('price', '?')}/night "
                f"| {addr.get('market', '')}, {addr.get('country', '')} "
                f"| score: {doc.get('score', 0):.4f}"
            )
