from config import OPENROUTER_API_KEY

if OPENROUTER_API_KEY:
    print("✅ OpenRouter API Key loaded successfully")
else:
    print("❌ OpenRouter API Key not found")