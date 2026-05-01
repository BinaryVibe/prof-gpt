from chromadb.utils import embedding_functions

miniLM_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

def ingest_chunks(chunks, collection):
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(ids=ids, documents=chunks)