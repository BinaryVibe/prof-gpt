import chromadb
from ml.rag.embedder import miniLM_ef

client = chromadb.PersistentClient(path="./data/chroma_db")
collection = client.get_or_create_collection(
    name="prof_gpt_materials",
    embedding_function=miniLM_ef
)

