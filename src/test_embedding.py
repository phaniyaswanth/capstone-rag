from openai import OpenAI
from config import OPENROUTER_API_KEY

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

response = client.embeddings.create(
    model="nvidia/nv-embedqa-e5-v5",
    input="Hello World"
)

embedding = response.data[0].embedding

print("Embedding length:", len(embedding))
print("First 5 values:", embedding[:5])
