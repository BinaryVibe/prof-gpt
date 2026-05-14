import json
import os
from backend.ml.rag.search import search_docs

def run_evaluation():
    # 1. Load the Evaluation Dataset
    eval_path = os.path.join(os.path.dirname(__file__), "eval_set.json")
    
    with open(eval_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    test_cases = data.get("test_cases", [])
    total_questions = len(test_cases)
    
    if total_questions == 0:
        print("Error: No test cases found in JSON.")
        return

    print(f"Starting Evaluation on {total_questions} queries...")
    
    hits = 0
    misses = []

    # 2. Run the Test Loop
    for idx, case in enumerate(test_cases):
        query = case["query"]
        expected_intent = case["intent"]
        expected_source = case["expected_source"]
        
        # We pass the expected_intent directly to purely test YOUR Track 1 RAG accuracy
        retrieved_blocks = search_docs(query=query, intent=expected_intent, k=3)
        
        # Extract the filenames from the search results
        retrieved_sources = [block["source"] for block in retrieved_blocks]
        
        # 3. Grade the Search
        if expected_source in retrieved_sources:
            hits += 1
        else:
            # Keep track of what we missed so we know HOW to tune it later
            misses.append({
                "query": query,
                "expected": expected_source,
                "retrieved": retrieved_sources
            })
            
    # 4. Calculate Final Metrics
    precision_at_3 = (hits / total_questions) * 100
    
    print("\n" + "="*40)
    print("EVALUATION RESULTS")
    print("="*40)
    print(f"Total Queries Tested: {total_questions}")
    print(f"Successful Retrievals (Hits): {hits}")
    print(f"Failed Retrievals (Misses): {total_questions - hits}")
    print(f"System Precision@3: {precision_at_3:.2f}%")
    
    if misses:
        print("\nTOP 3 MISSES (For Debugging):")
        for m in misses[:3]:
            print(f"- Q: {m['query']}\n  Expected: {m['expected']} | Got: {m['retrieved']}\n")

if __name__ == "__main__":
    run_evaluation()