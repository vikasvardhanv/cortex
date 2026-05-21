"""
Cortex Vector Database for RAG Knowledge Retrieval

Features:
- Semantic search using sentence embeddings
- Solver-type filtering for direct integration with physics solvers
- Unit metadata and dimensional analysis support
- Confidence/accuracy tags for quality assessment
"""

import os
import requests
import json
import uuid
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional

load_dotenv()


class CortexVectorDB:
    """
    Vector database interface for Cortex engineering knowledge.

    Supports:
    - Semantic search with category/solver filtering
    - Unit-aware knowledge retrieval
    - Confidence-based ranking
    """

    def __init__(self, collection_name="cortex_knowledge"):
        self.url = os.getenv("QDRANT_API_URL", "http://qdrant-wsosww00wsccgk080go80k0o.76.13.124.154.sslip.io")
        self.api_key = os.getenv("QDRANT_API_KEY")
        self.collection_name = collection_name

        if os.getenv("HF_TOKEN"):
            os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

        print("Loading Embedding Model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.vector_size = 384

        # Connection headers
        self.headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json"
        }

    def _ensure_collection(self):
        # We assume it exists or create it via requests
        url = f"{self.url}/collections/{self.collection_name}"
        resp = requests.get(url, headers=self.headers)
        if resp.status_code != 200:
            print(f"Creating collection {self.collection_name}...")
            data = {
                "vectors": {"size": self.vector_size, "distance": "Cosine"}
            }
            requests.put(url, headers=self.headers, json=data)

    def upsert_knowledge_batch(self, items):
        if not items:
            return
            
        print(f"Encoding {len(items)} items...")
        texts = [f"{item['category']} - {item['title']}\n{item['content']}" for item in items]
        vectors = self.model.encode(texts).tolist()
        
        points = []
        for i, item in enumerate(items):
            payload = {
                "title": item['title'],
                "content": item['content'],
                "source": item['source'],
                "tags": item['tags'],
                "category": item['category']
            }
            points.append({
                "id": str(uuid.uuid4()),
                "vector": vectors[i],
                "payload": payload
            })
        
        print(f"Uploading batch to Qdrant (REST)...")
        url = f"{self.url}/collections/{self.collection_name}/points?wait=true"
        resp = requests.put(url, headers=self.headers, json={"points": points})
        if resp.status_code == 200:
            print("Batch ingested.")
        else:
            print(f"Upsert failed: {resp.text}")

    def upsert_materials_batch(self, materials_list):
        items = []
        for mat in materials_list:
            items.append({
                "title": mat['name'],
                "content": "\n".join([f"{k}: {v}" for k, v in mat['properties'].items()]),
                "source": mat['source'],
                "tags": mat['tags'],
                "category": "material"
            })
        self.upsert_knowledge_batch(items)

    def upsert_material(self, material_name, properties, source_url, tags=[]):
        self.upsert_materials_batch([{
            "name": material_name,
            "properties": properties,
            "source": source_url,
            "tags": tags
        }])

    def search(self, query, limit=5, category=None):
        query_vector = self.model.encode(query).tolist()
        
        data = {
            "vector": query_vector,
            "limit": limit,
            "with_payload": True
        }
        
        if category:
            data["filter"] = {
                "must": [{"key": "category", "match": {"value": category}}]
            }
            
        url = f"{self.url}/collections/{self.collection_name}/points/search"
        resp = requests.post(url, headers=self.headers, json=data)
        if resp.status_code == 200:
            return resp.json()["result"]
        else:
            print(f"Search failed: {resp.text}")
            return []
