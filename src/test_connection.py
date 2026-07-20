from pymongo import MongoClient
from config import MONGODB_URI
import certifi

try:
    client = MongoClient(
        MONGODB_URI,
        tlsCAFile=certifi.where()
    )

    client.admin.command("ping")

    print("✅ Successfully connected to MongoDB Atlas!")

except Exception as e:
    print("❌ Connection failed")
    print(e)