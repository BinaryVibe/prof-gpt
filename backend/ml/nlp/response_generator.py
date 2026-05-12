import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def generate_academic_response(query, context_blocks, intent):
    """
    Combines NLP Intent with Document Context to generate a cited response.
    Expects context_blocks to be a list of dicts: [{"content": "...", "source": "..."}]
    """
    
    # 1. Format the context to enforce citations
    formatted_context = ""
    # Check if we got the expected list of dictionaries from search.py
    if isinstance(context_blocks, list):
        for block in context_blocks:
            source = block.get("source", "Unknown Source")
            content = block.get("content", "")
            formatted_context += f"Source: {source} - Content: \"{content}\"\n\n"
    else:
        # Fallback if a plain string is passed
        formatted_context = context_blocks

    # 2. Dynamic Prompting based on Intent
    if intent == "Policy":
        system_prompt = f"""
        You are 'Prof GPT', a strict academic assistant.
        The user is asking a policy question.
        
        INSTRUCTIONS:
        - Be extremely rigid. Quote exact numbers, penalties, and rules.
        - ALWAYS append a citation using the Source provided in the context (e.g., "[Source: syllabus.pdf]").
        - Use ONLY the provided context. If the answer isn't there, say "I'm sorry, I couldn't find that in the syllabus."
        
        COURSE CONTEXT:
        {formatted_context}
        """
    elif intent == "Technical":
        system_prompt = f"""
        You are 'Prof GPT', a helpful and patient university professor.
        The user is asking a technical concept question.
        
        INSTRUCTIONS:
        - Explain the concept clearly. Use helpful analogies if appropriate to ensure the student understands.
        - ALWAYS append a citation using the Source provided in the context (e.g., "[Source: lecture_1.pdf]").
        - Use ONLY the provided context. If the answer isn't there, say "I'm sorry, I couldn't find that in the materials."
        
        COURSE CONTEXT:
        {formatted_context}
        """
    else:
        # Default for Schedule or other intents
        system_prompt = f"""
        You are 'Prof GPT', a helpful academic assistant.
        The user is asking a '{intent}' question.
        
        INSTRUCTIONS:
        - Answer clearly and concisely.
        - ALWAYS append a citation using the Source provided in the context.
        - Use ONLY the provided context.
        
        COURSE CONTEXT:
        {formatted_context}
        """

    # 3. Generate Content
    response = client.models.generate_content(
        model="gemini-2.5-flash", 
        contents=query,
        config={'system_instruction': system_prompt}
    )
    
    return response.text

if __name__ == "__main__":
    dummy_query = "What happens if I turn my assignment in late?"
    
    # Mocking the new list-of-dicts structure that Ayaan's search_docs() returns
    dummy_context_blocks = [
        {
            "content": "Late assignments will receive a 10% penalty per day.",
            "source": "comsats_syllabus_2026.pdf"
        }
    ]
    
    intent_from_ml = "Policy"
    
    print("--- PROF GPT SAYS ---")
    print(generate_academic_response(dummy_query, dummy_context_blocks, intent_from_ml))