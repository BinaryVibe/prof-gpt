from backend.database import client
from backend.ml.rag.embedder import miniLM_ef

collection = client.get_or_create_collection(
    name="prof_gpt_materials", 
    embedding_function=miniLM_ef
)

def search_docs(query: str, intent: str=None, k:int=3):
    intent_map = {
    "Policy": "policy",
    "Technical": "technical",
    "Schedule": "schedule"    }
    if intent in intent_map:
        where_filter = {"category": {"$eq": intent_map[intent]}}
    else:
        where_filter = None
        
    results = collection.query(
        query_texts=[query],
        include=["documents", "metadatas", "distances"],
        where=where_filter,
        n_results=k
    )

    docs = results['documents'][0]
    dists= results['distances'][0]
    metas = results['metadatas'][0] 


    context_blocks = []
    for doc, meta, dist in zip(docs, metas, dists):
        print(f"\n Document: {doc}\nDistance: {dist}")

        block = {
            "content": doc,
            "source": meta['source'],
            "distance": dist
        }
        context_blocks.append(block)
    return context_blocks

# TEST BLOCK (Run this file directly to test)
if __name__ == "__main__":
    # Note: This assumes you have already run your embedder.py to ingest some dummy text into your local database!
    
    test_question = "What is the penalty for late assignment submissions?"
    test_intent = "Policy"
    print("TESTING K=1 (Top Match)")
    search_docs(test_question, intent=test_intent, k=1)
    
    print("\nTESTING K=3 (Broad Search)")
    search_docs(test_question, k=3)
