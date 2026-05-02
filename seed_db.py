from backend.database import client
from backend.ml.rag.embedder import miniLM_ef, ingest_chunks
from backend.ml.rag.chunker import split_document

# 1. Connect to your database
collection = client.get_or_create_collection(
    name="prof_gpt_materials", 
    embedding_function=miniLM_ef
)

# 2. Some dummy syllabus data
dummy_syllabus = """
COMSATS University Late Policy:
Any assignment submitted after the official deadline will incur a strict 10% penalty per day. 
After 3 days, the assignment will absolutely not be accepted and the student will receive a zero.

Final Project Requirements:
The final project must be a full-stack web application. 
Students must deploy the frontend to Vercel and the backend to a Linux server.
"""

print("Chunking text...")
# 3. Slice the text using your LangChain logic
chunks = split_document(dummy_syllabus)

print(f"Created {len(chunks)} chunks. Vectorizing and saving to ChromaDB...")
# 4. Turn text into math and save to the local file system
ingest_chunks(chunks, collection)

print("✅ Dummy data successfully embedded and saved!")