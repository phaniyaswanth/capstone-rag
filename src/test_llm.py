from openai import OpenAI
from config import OPENROUTER_API_KEY

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

response = client.chat.completions.create(
    model="openrouter/free",
    messages=[
        {
            "role": "user",
            "content": "Say hello in one sentence."
        }
    ]
)

print(response.choices[0].message.content)