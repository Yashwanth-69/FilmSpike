"""
tmdb_ingest.py
Phase 2 — One-time script to fetch movies from TMDB API and ingest into ChromaDB.

Usage:
    python -m src.analysis.tmdb_ingest --pages 20
    python -m src.analysis.tmdb_ingest --pages 20 --top-rated

Requires: TMDB_API_KEY in .env file
"""
import os
import sys
import logging
import argparse
import csv
import time
from pathlib import Path
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent.parent
_FALLBACK_CSV = _ROOT / "data" / "movies_fallback.csv"
_DB_PATH = _ROOT / "data" / "chroma_db"
_COLLECTION_NAME = "movies"

def load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv(_ROOT / ".env")
    except ImportError:
        pass
    return os.getenv("TMDB_API_KEY")


def fetch_tmdb_movies(api_key: str, pages: int = 20, endpoint: str = "popular") -> List[Dict]:
    """Fetch movies from TMDB API."""
    import requests
    BASE = "https://api.themoviedb.org/3"
    GENRE_MAP = {
        28: "Action", 12: "Adventure", 16: "Animation", 35: "Comedy",
        80: "Crime", 99: "Documentary", 18: "Drama", 10751: "Family",
        14: "Fantasy", 36: "History", 27: "Horror", 10402: "Music",
        9648: "Mystery", 10749: "Romance", 878: "Sci-Fi", 10770: "TV Movie",
        53: "Thriller", 10752: "War", 37: "Western",
    }

    all_movies = []
    seen_ids = set()

    for page in range(1, pages + 1):
        url = f"{BASE}/movie/{endpoint}"
        resp = requests.get(url, params={"api_key": api_key, "page": page}, timeout=10)
        if resp.status_code != 200:
            logger.warning(f"TMDB API error page {page}: {resp.status_code}")
            break

        data = resp.json()
        for m in data.get("results", []):
            if m["id"] in seen_ids:
                continue
            seen_ids.add(m["id"])

            # Fetch extended details for revenue
            detail_resp = requests.get(
                f"{BASE}/movie/{m['id']}",
                params={"api_key": api_key},
                timeout=10,
            )
            revenue = 0
            if detail_resp.status_code == 200:
                revenue = detail_resp.json().get("revenue", 0) or 0

            genres_str = ", ".join(GENRE_MAP.get(gid, "") for gid in m.get("genre_ids", []) if gid in GENRE_MAP)
            year = m.get("release_date", "")[:4] if m.get("release_date") else ""

            all_movies.append({
                "tmdb_id":      m["id"],
                "title":        m.get("title", ""),
                "overview":     m.get("overview", ""),
                "genres":       genres_str,
                "release_date": m.get("release_date", ""),
                "year":         year,
                "revenue":      revenue,
                "vote_average": m.get("vote_average", 0.0),
                "vote_count":   m.get("vote_count", 0),
            })

        logger.info(f"Page {page}/{pages} fetched — {len(all_movies)} movies so far")
        time.sleep(0.25)  # rate limit

    return all_movies


def save_to_csv(movies: List[Dict], path: Path):
    """Save movies to CSV (usable as fallback)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["tmdb_id", "title", "overview", "genres", "release_date", "year", "revenue", "vote_average", "vote_count"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(movies)
    logger.info(f"Saved {len(movies)} movies to {path}")


def ingest_to_chromadb(movies: List[Dict]):
    """Encode movies with CLIP and store in ChromaDB."""
    import torch
    import clip as clip_module
    import chromadb

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading CLIP text encoder on {device}...")
    model, _ = clip_module.load("ViT-B/32", device=device)
    model.eval()

    _DB_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(_DB_PATH))
    collection = client.get_or_create_collection(
        name=_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids, embeddings, documents, metadatas = [], [], [], []
    BATCH = 32

    for i in range(0, len(movies), BATCH):
        batch = movies[i : i + BATCH]
        texts = [f"{m['genres']}: {m['overview']}" or m["title"] for m in batch]

        tokens = clip_module.tokenize(texts, truncate=True).to(device)
        with torch.no_grad():
            feats = model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            feats_np = feats.cpu().numpy()

        for j, m in enumerate(batch):
            ids.append(f"tmdb_{m['tmdb_id']}")
            embeddings.append(feats_np[j].tolist())
            documents.append(texts[j])
            metadatas.append({
                "title":        m["title"],
                "year":         int(m["year"]) if m["year"] else 0,
                "genres":       m["genres"],
                "revenue":      int(m["revenue"]) if m["revenue"] else 0,
                "release_date": m["release_date"],
                "vote_average": float(m["vote_average"]) if m["vote_average"] else 0.0,
                "tmdb_id":      int(m["tmdb_id"]),
            })

        logger.info(f"Encoded {min(i+BATCH, len(movies))}/{len(movies)} movies")

    # Upsert in batches of 100
    for i in range(0, len(ids), 100):
        collection.upsert(
            ids=ids[i:i+100],
            embeddings=embeddings[i:i+100],
            documents=documents[i:i+100],
            metadatas=metadatas[i:i+100],
        )

    logger.info(f"ChromaDB now has {collection.count()} movies")


def main():
    parser = argparse.ArgumentParser(description="Ingest TMDB movies into ChromaDB")
    parser.add_argument("--pages", type=int, default=20, help="Number of pages to fetch (20 pages ≈ 400 movies)")
    parser.add_argument("--top-rated", action="store_true", help="Fetch top-rated instead of popular")
    parser.add_argument("--csv-only", action="store_true", help="Only save CSV, skip ChromaDB")
    parser.add_argument("--from-csv", type=str, help="Ingest from existing CSV (skip TMDB API)")
    args = parser.parse_args()

    if args.from_csv:
        import csv as csv_mod
        with open(args.from_csv, newline="", encoding="utf-8") as f:
            movies = list(csv_mod.DictReader(f))
        ingest_to_chromadb(movies)
        return

    api_key = load_env()
    if not api_key:
        logger.error("TMDB_API_KEY not found in .env — cannot fetch from TMDB API.")
        logger.info(f"Tip: set TMDB_API_KEY in .env or use --from-csv {_FALLBACK_CSV}")
        sys.exit(1)

    endpoint = "top_rated" if args.top_rated else "popular"
    logger.info(f"Fetching {args.pages} pages of {endpoint} movies from TMDB...")
    movies = fetch_tmdb_movies(api_key, pages=args.pages, endpoint=endpoint)

    # Always update the fallback CSV
    save_to_csv(movies, _FALLBACK_CSV)

    if not args.csv_only:
        ingest_to_chromadb(movies)


if __name__ == "__main__":
    main()
