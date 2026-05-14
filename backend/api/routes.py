import joblib
import os
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from backend.ml.rag.search import search_docs
from backend.ml.nlp.response_generator import generate_academic_response

router = APIRouter()

MODEL_PATH = "backend/ml/intent/intent_model.pkl"
intent_pipeline = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

class ChatRequest(BaseModel):
    query: str

def get_predicted_intent(query: str):
    if intent_pipeline:
        return intent_pipeline.predict([query])[0]
    return "General"

@router.post("/ask")
async def ask_prof_gpt(request: ChatRequest):
    user_query = request.query
    
    detected_intent = get_predicted_intent(user_query)
    
    retrieved_context = search_docs(query=user_query, intent=detected_intent, k=3)
    
    final_answer = generate_academic_response(
        query=user_query, 
        context_blocks=retrieved_context, 
        intent=detected_intent
    )
    
    return {
        "answer": final_answer,
        "intent": detected_intent,
        "sources": list(set([block["source"] for block in retrieved_context]))
    }