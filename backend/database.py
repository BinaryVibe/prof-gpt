import chromadb
client = chromadb.PersistentClient(path="./data/chroma_db")
collection = client.get_or_create_collection(name="prof_gpt_materials")

