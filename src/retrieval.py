from pymongo import MongoClient
import certifi
from openai import OpenAI
from config import (
    MONGODB_URI,
    OPENROUTER_API_KEY,
    EMBEDDING_MODEL,
    SOURCE_DB,
    RAG_CHUNKS_COLLECTION,
)

client = MongoClient(MONGODB_URI, tlsCAFile=certifi.where())
db = client[SOURCE_DB]
collection = db["listingsAndReviews"]
chunks_col = db[RAG_CHUNKS_COLLECTION]

openai_client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)


def get_listing_by_name(name):
    return collection.find_one({"name": name})


def get_listing_by_id(listing_id):
    return collection.find_one({"_id": listing_id})


def search_by_country(country):
    return list(collection.find({"address.country": country}).limit(5))


def search_by_property_type(property_type):
    return list(collection.find({"property_type": property_type}).limit(5))


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
    print("----- Country Search -----")
    listings = search_by_country("Portugal")
    for listing in listings:
        print(listing["name"])

    print("\n----- Vector Search -----")
    results = vector_search("quiet apartment near the beach with wifi", k=3)
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.get('name')} | score: {doc.get('score', 0):.4f}")