from fastapi import APIRouter
from pydantic import BaseModel
from backend.ml.nlp.classifier import predict_intent
from backend.ml.nlp.response_generator import generate_academic_response

router = APIRouter()

class ChatRequest(BaseModel):
    query: str

@router.post("/ask")
async def ask_prof_gpt(request: ChatRequest):
    user_query = request.query
    
    real_intent = predict_intent(user_query)
    
    # 2. Ayaan yahan replace karna
    mock_context = "Late assignments receive a 10% penalty per day. The midterm is on Friday."
    
    # 3. Generate response using your ML-detected intent
    final_answer = generate_academic_response(
        query=user_query, 
        context=mock_context, 
        intent=real_intent
    )
    
    return {
        "status": "success",
        "intent_detected": real_intent, # This now shows what YOUR model thought!
        "response": final_answer
    }