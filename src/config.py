import os
from dotenv import load_dotenv

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")

EMBEDDING_MODEL = "openai/text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536

SOURCE_DB = "sample_airbnb"
SOURCE_COLLECTION = "listingsAndReviews"
RAG_CHUNKS_COLLECTION = "rag_chunks"

RAG_MODE = os.getenv("RAG_MODE", "baseline")