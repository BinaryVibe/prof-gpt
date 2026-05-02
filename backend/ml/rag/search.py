from backend.database import client
from backend.ml.rag.embedder import miniLM_ef

collection = client.get_or_create_collection(
    name="prof_gpt_materials", 
    embedding_function=miniLM_ef
)

def search_docs(query: str, k:int=3):
    results = collection.query(
        query_texts=[query],
        n_results=k
    )

    docs = results['documents'][0]
    dists= results['distances'][0]

    for doc, dist in zip(docs, dists):
        print(f"\n Document: {doc}\nDistance: {dist}")

    return docs

# TEST BLOCK (Run this file directly to test)
if __name__ == "__main__":
    # Note: This assumes you have already run your embedder.py to ingest some dummy text into your local database!
    
    test_question = "What is the penalty for late assignment submissions?"
    
    print("TESTING K=1 (Top Match)")
    search_docs(test_question, k=1)
    
    print("\nTESTING K=3 (Broad Search)")
    search_docs(test_question, k=3)
