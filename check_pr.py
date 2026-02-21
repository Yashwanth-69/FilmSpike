import chromadb
client = chromadb.PersistentClient(path="data/chroma_db")
collection = client.get_collection("movies")
res = collection.get(where={"title": "Pacific Rim"})
print("Pacific Rim in DB metadata:", res["metadatas"])
print("IDs:", res["ids"])
