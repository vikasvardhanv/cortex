
import os
from knowledge.rag.vector_db import CortexVectorDB

def verify_knowledge():
    db = CortexVectorDB()
    
    print("\n--- Verifying Knowledge Base Retrieval ---")
    
    # Query 1: Thermal conductivity
    print("\nQuery 1: 'High thermal conductivity metals for rocket nozzles'")
    results = db.search("High thermal conductivity metals for rocket nozzles", limit=3)
    for i, res in enumerate(results):
        payload = res.get('payload', {})
        print(f"{i+1}. {payload.get('title') or payload.get('name')} (Score: {res.get('score'):.4f})")
        
    # Query 2: Physics equations
    print("\nQuery 2: 'How to calculate thrust of a rocket?'")
    results = db.search("How to calculate thrust of a rocket?", limit=2)
    for i, res in enumerate(results):
        payload = res.get('payload', {})
        print(f"{i+1}. {payload.get('title')} (Score: {res.get('score'):.4f})")
        print(f"   Content: {payload.get('content')}")

    # Query 3: Dimensionless numbers
    print("\nQuery 3: 'What is Reynolds Number?'")
    results = db.search("What is Reynolds Number?", limit=1)
    for i, res in enumerate(results):
        payload = res.get('payload', {})
        print(f"{i+1}. {payload.get('title')} (Score: {res.get('score'):.4f})")
        print(f"   Content: {payload.get('content')}")

if __name__ == "__main__":
    verify_knowledge()
