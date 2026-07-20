import argparse
import time

from pymongo import MongoClient
from openai import OpenAI
import certifi

from config import (
    MONGODB_URI,
    OPENROUTER_API_KEY,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSIONS,
    SOURCE_DB,
    SOURCE_COLLECTION,
    RAG_CHUNKS_COLLECTION,
)

mongo_client = MongoClient(MONGODB_URI, tlsCAFile=certifi.where())
db = mongo_client[SOURCE_DB]
source = db[SOURCE_COLLECTION]
chunks_col = db[RAG_CHUNKS_COLLECTION]

openai_client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)


def build_chunk_text(doc: dict) -> str:
    parts = []

    name = doc.get("name", "")
    if name:
        parts.append(f"Listing: {name}")

    prop_type = doc.get("property_type", "")
    room_type = doc.get("room_type", "")
    if prop_type or room_type:
        parts.append(f"Property Type: {prop_type} | Room Type: {room_type}")

    address = doc.get("address", {})
    location_parts = [address.get(k, "") for k in ("street", "market", "country") if address.get(k)]
    if location_parts:
        parts.append(f"Location: {', '.join(location_parts)}")

    for field, label in [
        ("summary", "Summary"),
        ("description", "Description"),
        ("space", "Space"),
        ("neighborhood_overview", "Neighborhood"),
        ("transit", "Transit"),
    ]:
        value = doc.get(field, "")
        if value and value.strip():
            parts.append(f"\n{label}: {value.strip()}")

    amenities = doc.get("amenities", [])
    if amenities:
        parts.append(f"\nAmenities: {', '.join(amenities)}")

    return "\n".join(parts)


def build_metadata(doc: dict) -> dict:
    address = doc.get("address", {})

    def to_float(val):
        if val is None:
            return None
        try:
            return float(str(val))
        except (ValueError, TypeError):
            return None

    return {
        "listing_id": str(doc["_id"]),
        "name": doc.get("name", ""),
        "listing_url": doc.get("listing_url", ""),
        "property_type": doc.get("property_type", ""),
        "room_type": doc.get("room_type", ""),
        "price": to_float(doc.get("price")),
        "bedrooms": doc.get("bedrooms"),
        "beds": doc.get("beds"),
        "bathrooms": to_float(doc.get("bathrooms")),
        "accommodates": doc.get("accommodates"),
        "address": {
            "street": address.get("street", ""),
            "market": address.get("market", ""),
            "country": address.get("country", ""),
            "country_code": address.get("country_code", ""),
        },
        "review_scores": doc.get("review_scores", {}),
    }


def embed_batch(texts: list[str]) -> list[list[float]]:
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    sorted_data = sorted(response.data, key=lambda x: x.index)
    return [item.embedding for item in sorted_data]


def ingest(limit: int | None = 200):
    print(f"Starting ingest | model: {EMBEDDING_MODEL} | limit: {'all' if limit is None else limit}")

    chunks_col.drop()

    query = {"description": {"$ne": "", "$exists": True}}
    cursor = source.find(query)
    if limit is not None:
        cursor = cursor.limit(limit)

    listings = list(cursor)
    print(f"Loaded {len(listings)} listings")

    if not listings:
        print("No listings found. Check your MongoDB connection.")
        return

    chunk_texts = []
    metadata_list = []
    skipped = 0

    for doc in listings:
        text = build_chunk_text(doc)
        if not text.strip():
            skipped += 1
            continue
        chunk_texts.append(text)
        metadata_list.append(build_metadata(doc))

    if skipped:
        print(f"Skipped {skipped} empty documents")

    BATCH_SIZE = 100
    all_embeddings = []
    total_batches = (len(chunk_texts) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(chunk_texts), BATCH_SIZE):
        batch = chunk_texts[i : i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        print(f"Embedding batch {batch_num}/{total_batches} ({len(batch)} texts)")

        try:
            embeddings = embed_batch(batch)
            all_embeddings.extend(embeddings)
        except Exception as e:
            print(f"Batch {batch_num} failed: {e}. Retrying after 5s...")
            time.sleep(5)
            try:
                embeddings = embed_batch(batch)
                all_embeddings.extend(embeddings)
            except Exception as e2:
                print(f"Retry failed: {e2}. Aborting.")
                return

        if batch_num < total_batches:
            time.sleep(0.5)

    docs_to_insert = [
        {"text": text, "embedding": embedding, **meta}
        for text, embedding, meta in zip(chunk_texts, all_embeddings, metadata_list)
    ]

    INSERT_BATCH = 500
    inserted = 0
    for i in range(0, len(docs_to_insert), INSERT_BATCH):
        result = chunks_col.insert_many(docs_to_insert[i : i + INSERT_BATCH])
        inserted += len(result.inserted_ids)

    print(f"Done. Inserted {inserted} documents into {RAG_CHUNKS_COLLECTION} ({EMBEDDING_DIMENSIONS}d embeddings)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Embed all listings")
    parser.add_argument("--limit", type=int, default=200, help="Number of listings to embed (default: 200)")
    args = parser.parse_args()

    ingest(limit=None if args.all else args.limit)