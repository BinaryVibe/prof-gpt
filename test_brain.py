# Save this as test_brain.py or run it in a python shell
import joblib
pipeline = joblib.load("backend/ml/intent/intent_model.pkl")

test_queries = [
    "How do I apply for the Shine internship?", # Career_cdc
    "What is the late fee for sessional marks?", # Evaluation
    "Explain the logic of a hash map."          # Technical
]

for q in test_queries:
    print(f"Query: {q} -> Predicted: {pipeline.predict([q])[0]}")