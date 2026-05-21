import os
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()

url = os.getenv("QDRANT_API_URL")
api_key = os.getenv("QDRANT_API_KEY")

print(f"Testing connection to {url}")
try:
    client = QdrantClient(url=url, api_key=api_key, timeout=10, prefer_grpc=False)
    collections = client.get_collections()
    print(f"Success! Collections: {collections}")
except Exception as e:
    print(f"Failed with URL: {e}")

hostname = url.replace("http://", "").split(":")[0]
print(f"Testing connection to host={hostname}, port=80")
try:
    client = QdrantClient(host=hostname, port=80, api_key=api_key, timeout=10, prefer_grpc=False)
    collections = client.get_collections()
    print(f"Success with host/port! Collections: {collections}")
except Exception as e:
    print(f"Failed with host/port: {e}")
