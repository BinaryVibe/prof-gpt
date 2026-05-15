import joblib
import os
import io
import fitz
from fastapi import APIRouter, Depends, File, UploadFile
from pydantic import BaseModel
from typing import List

# RAG & NLP Imports
from backend.ml.rag.search import search_docs
from backend.ml.nlp.response_generator import generate_academic_response

# Embedder & Database Imports
from backend.database import client
from backend.ml.rag.embedder import miniLM_ef, ingest_chunks
from backend.ml.rag.chunker import split_document

router = APIRouter()

# --- 1. Intent Model Setup ---
MODEL_PATH = "backend/ml/intent/intent_model.pkl"
intent_pipeline = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

class ChatRequest(BaseModel):
    query: str

def get_predicted_intent(query: str):
    if intent_pipeline:
        return intent_pipeline.predict([query])[0]
    return "General"

# --- 2. The Chat Endpoint ---
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

@router.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    collection = client.get_or_create_collection(
        name="prof_gpt_materials",
        embedding_function=miniLM_ef
    )
    
    processed_count = 0
    for file in files:
        contents = await file.read()
        
        # Open the PDF directly from the byte stream using PyMuPDF
        doc = fitz.open(stream=contents, filetype="pdf")
        
        text = ""
        for page in doc:
            extracted = page.get_text("text")
            if extracted:
                text += extracted + "\n"
                
        # Chunk and ingest the text
        if text.strip():
            chunks = split_document(text)
            ingest_chunks(chunks, collection, source_name=file.filename, category="technical")
            processed_count += 1
            print(f" Ingested {file.filename}: {len(chunks)} chunks.") # Debug print

    return {"message": f"Successfully processed {processed_count} documents"}