import os
from google import genai
from dotenv import load_dotenv

# Load the secret key from .env
load_dotenv()

# Connect to Gemini
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def generate_academic_response(query, context, intent):
    """
    This combines your NLP Intent (Track 2) with the Document Context (Track 1).
    """
    system_prompt = f"""
    You are 'Prof GPT', a helpful academic assistant.
    The user is asking a '{intent}' question.
    Use ONLY the context provided to answer. If you cannot find the answer in the context, 
    say "I'm sorry, I couldn't find that in the syllabus."
    
    COURSE CONTEXT:
    {context}
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash", 
        contents=query,
        config={'system_instruction': system_prompt}
    )
    
    return response.text

if __name__ == "__main__":
    dummy_query = "What happens if I turn my assignment in late?"
    dummy_context = "Late assignments will receive a 10% penalty per day."
    intent_from_ml = "Policy"
    
    print("--- PROF GPT SAYS ---")
    print(generate_academic_response(dummy_query, dummy_context, intent_from_ml))