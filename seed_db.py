from backend.database import client
from backend.ml.rag.embedder import miniLM_ef, ingest_chunks
from backend.ml.rag.chunker import split_document

# 1. Connect to your database
collection = client.get_or_create_collection(
    name="prof_gpt_materials", 
    embedding_function=miniLM_ef
)

print("Chunking and embedding Policy data...")
# 2. Dummy Syllabus Data (Category: policy)
dummy_syllabus = """
COMSATS University Late Policy:
Any assignment submitted after the official deadline will incur a strict 10% penalty per day. 
After 3 days, the assignment will absolutely not be accepted and the student will receive a zero.
"""
syllabus_chunks = split_document(dummy_syllabus)
ingest_chunks(syllabus_chunks, collection, source_name="comsats_syllabus_2026.pdf", category="policy")

print("Chunking and embedding Technical data...")
# 3. Dummy Lecture Data (Category: technical)
dummy_lecture = """
Introduction to React:
React is a component-based frontend library. You manage state using hooks like useState and useEffect.
To update the UI, React uses a Virtual DOM.
"""
lecture_chunks = split_document(dummy_lecture)
ingest_chunks(lecture_chunks, collection, source_name="react_lecture_1.pdf", category="technical")

print("✅ Dummy data successfully embedded and saved!")