import json
import numpy as np
import chromadb
from src.analysis.keyword_aggregator import KeywordAggregator
from src.analysis.movie_matcher import MovieMatcher

with open("data/output/test.json", "r") as f:
    fingerprint = json.load(f)

aggregator = KeywordAggregator()
kw_result = aggregator.aggregate(fingerprint)
film_vector = kw_result.get("film_vector")
top_keywords = kw_result.get("top_keywords", [])

matcher = MovieMatcher()
client = chromadb.PersistentClient(path="data/chroma_db")
collection = client.get_collection("movies")

# Get PR
pr = collection.get(ids=["tmdb_68726"], include=["embeddings", "metadatas"])
pr_emb = np.array(pr["embeddings"][0])
meta = pr["metadatas"][0]

film_vec_np = np.array(film_vector)

import torch, clip
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _ = clip.load("ViT-B/32", device=device)
model.eval()
kw_text = ", ".join(top_keywords[:10])
tokens = clip.tokenize([kw_text], truncate=True).to(device)
with torch.no_grad():
    kw_vec = model.encode_text(tokens)
    kw_vec = kw_vec / kw_vec.norm(dim=-1, keepdim=True)
    kw_vec_np = kw_vec.cpu().numpy()[0]

norm_emb = np.linalg.norm(pr_emb)
norm_film = np.linalg.norm(film_vec_np)
norm_kw = np.linalg.norm(kw_vec_np)

v_sim = (float(np.dot(pr_emb, film_vec_np) / (norm_emb * norm_film)) + 1.0) / 2.0
k_sim = (float(np.dot(pr_emb, kw_vec_np) / (norm_emb * norm_kw)) + 1.0) / 2.0
blended = 0.6 * v_sim + 0.4 * k_sim

print(f"Pacific Rim Exact Scores -> Vis: {v_sim:.4f}, Kw: {k_sim:.4f}, Blended: {blended:.4f}")

# Check ranks
img_res = collection.query(query_embeddings=[film_vec_np.tolist()], n_results=500)
try:
    img_rank = img_res["ids"][0].index("tmdb_68726") + 1
    print(f"Vis Rank: {img_rank}")
except ValueError:
    print("Vis Rank: >500")

kw_res = collection.query(query_embeddings=[kw_vec_np.tolist()], n_results=500)
try:
    kw_rank = kw_res["ids"][0].index("tmdb_68726") + 1
    print(f"Kw Rank: {kw_rank}")
except ValueError:
    print("Kw Rank: >500")

