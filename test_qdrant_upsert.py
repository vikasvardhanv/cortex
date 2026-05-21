import os
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv
import uuid

load_dotenv()

url = os.getenv("QDRANT_API_URL")
api_key = os.getenv("QDRANT_API_KEY")

client = QdrantClient(url=url, api_key=api_key, timeout=30, prefer_grpc=False)

print("Attempting to upsert 1 dummy point...")
try:
    client.upsert(
        collection_name="cortex_knowledge",
        points=[
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=[0.1] * 384,
                payload={"test": "data"}
            )
        ]
    )
    print("Success!")
except Exception as e:
    print(f"Failed: {e}")
