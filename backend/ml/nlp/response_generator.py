import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def generate_academic_response(query, context_blocks, intent):
    formatted_context = ""
    if isinstance(context_blocks, list):
        for block in context_blocks:
            source = block.get("source", "Unknown Source")
            content = block.get("content", "")
            formatted_context += f"Source: {source} - Content: \"{content}\"\n\n"
    else:
        formatted_context = context_blocks

    persona_map = {
        "Technical": "patient computer science professor using helpful analogies",
        "Lab_manual": "strict lab instructor emphasizing safety and exact steps",
        "Policy": "rigid academic advisor quoting exact university rules",
        "Evaluation": "clear and precise grading coordinator",
        "Schedule": "helpful university registrar assistant",
        "Admin_fees": "empathetic but firm financial aid counselor",
        "Past_papers": "encouraging senior student sharing study patterns",
        "Career_cdc": "motivational Career Development Center (CDC) mentor",
        "Society_events": "enthusiastic campus life coordinator",
        "Clubs_acm": "high-energy tech community leader"
    }

    persona = persona_map.get(intent, "helpful academic assistant")

    system_prompt = f"""
    You are 'Prof GPT', a {persona} at COMSATS University Wah Campus.
    The user is asking a question related to: {intent}.
    
    INSTRUCTIONS:
    - Answer clearly, matching your persona style.
    - ALWAYS append a citation using the Source provided in the context (e.g., "[Source: file.pdf]").
    - Use ONLY the provided context. If the answer isn't there, say "I couldn't find that in the official materials."
    
    COURSE CONTEXT:
    {formatted_context}
    """

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite", 
        contents=query,
        config={'system_instruction': system_prompt}
    )
    
    return response.text