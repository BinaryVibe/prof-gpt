from chromadb.utils import embedding_functions

miniLM_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

def ingest_chunks(chunks, collection, source_name="unknown.pdf"):
    ids = [f"{source_name}_chunk_{i}" for i in range(len(chunks))]
    
    metas = [{"source": source_name} for _ in range(len(chunks))]
    
    collection.add(
        ids=ids, 
        documents=chunks,
        metadatas=metas
    )