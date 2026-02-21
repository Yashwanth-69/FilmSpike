"""
tmdb5000_ingest.py
Ingest the Kaggle TMDB 5000 Movies dataset into ChromaDB using CLIP text embeddings.

Dataset: dataset/tmdb_5000_movies.csv  (4803 movies)
No API key required — fully offline.

Usage:
    python -m src.analysis.tmdb5000_ingest
    python -m src.analysis.tmdb5000_ingest --limit 1000   # ingest only first 1000
    python -m src.analysis.tmdb5000_ingest --reset        # wipe ChromaDB first
"""
import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent.parent
_CSV   = _ROOT / "dataset" / "tmdb_5000_movies.csv"
_DB_PATH = _ROOT / "data" / "chroma_db"
_COLLECTION = "movies"


def parse_genres(raw: str) -> str:
    """Convert '[{"id":28,"name":"Action"}, ...]' → 'Action, Sci-Fi'"""
    try:
        items = json.loads(raw)
        return ", ".join(item["name"] for item in items if "name" in item)
    except Exception:
        return ""


def parse_keywords(raw: str, max_keywords: int = 15) -> str:
    """Convert TMDB keywords JSON array → 'giant robot, giant monster, ...'"""
    try:
        items = json.loads(raw)
        return ", ".join(item["name"] for item in items[:max_keywords] if "name" in item)
    except Exception:
        return ""


def build_embed_text(genres: str, keywords: str, overview: str) -> str:
    """
    Build rich text for CLIP encoding.
    Format: "Action, Sci-Fi | Keywords: giant robot, giant monster | Overview: ..."
    Keywords get leading position so CLIP treats them as high-signal.
    """
    parts = []
    if genres:
        parts.append(genres)
    if keywords:
        parts.append(f"Keywords: {keywords}")
    if overview:
        parts.append(f"Overview: {overview}")
    return " | ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Ingest TMDB 5000 dataset into ChromaDB")
    parser.add_argument("--limit", type=int, default=None, help="Max movies to ingest (default: all 4803)")
    parser.add_argument("--reset", action="store_true", help="Delete existing ChromaDB collection before ingesting")
    parser.add_argument("--batch", type=int, default=32, help="CLIP encoding batch size (default: 32)")
    args = parser.parse_args()

    if not _CSV.exists():
        logger.error(f"Dataset not found at {_CSV}")
        sys.exit(1)

    # ── Load dataset ──────────────────────────────────────────────────────────
    import pandas as pd
    df = pd.read_csv(_CSV)
    logger.info(f"Loaded {len(df)} movies from {_CSV.name}")

    # Drop rows with missing overview (can't embed them meaningfully)
    df = df[df["overview"].notna() & (df["overview"].str.strip() != "")]
    logger.info(f"  → {len(df)} movies after filtering empty overviews")

    if args.limit:
        df = df.head(args.limit)
        logger.info(f"  → Limited to {len(df)} movies")

    # ── ChromaDB setup ────────────────────────────────────────────────────────
    import chromadb
    _DB_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(_DB_PATH))

    if args.reset:
        try:
            client.delete_collection(_COLLECTION)
            logger.info("Existing collection deleted")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    existing = collection.count()
    logger.info(f"ChromaDB collection '{_COLLECTION}' — {existing} existing entries")

    # ── Load CLIP ─────────────────────────────────────────────────────────────
    import torch
    import clip as clip_module

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading CLIP ViT-B/32 text encoder on {device}...")
    model, _ = clip_module.load("ViT-B/32", device=device)
    model.eval()

    # ── Encode & ingest in batches ────────────────────────────────────────────
    ids, embeddings, documents, metadatas = [], [], [], []
    total = len(df)

    for start in range(0, total, args.batch):
        batch_df = df.iloc[start : start + args.batch]
        texts = []

        for _, row in batch_df.iterrows():
            genres   = parse_genres(str(row.get("genres", "[]")))
            keywords = parse_keywords(str(row.get("keywords", "[]")))
            overview = str(row.get("overview", "")).strip()
            text = build_embed_text(genres, keywords, overview) or str(row.get("title", "unknown"))
            texts.append(text)

            year = str(row.get("release_date", ""))[:4]
            ids.append(f"tmdb_{int(row['id'])}")
            documents.append(text)
            metadatas.append({
                "title":        str(row.get("title", "")),
                "year":         int(year) if year.isdigit() else 0,
                "genres":       genres,
                "keywords":     keywords,          # store for dual-query text matching
                "revenue":      int(row.get("revenue", 0) or 0),
                "release_date": str(row.get("release_date", "")),
                "vote_average": float(row.get("vote_average", 0.0) or 0.0),
                "tmdb_id":      int(row["id"]),
            })

        # CLIP text encode
        tokens = clip_module.tokenize(texts, truncate=True).to(device)
        with torch.no_grad():
            feats = model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            feats_np = feats.cpu().numpy()

        embeddings.extend(feats_np.tolist())

        done = min(start + args.batch, total)
        logger.info(f"  Encoded {done}/{total} movies...")

    # Upsert into ChromaDB in chunks of 500
    chunk = 500
    for i in range(0, len(ids), chunk):
        collection.upsert(
            ids=ids[i:i+chunk],
            embeddings=embeddings[i:i+chunk],
            documents=documents[i:i+chunk],
            metadatas=metadatas[i:i+chunk],
        )
        logger.info(f"  Upserted {min(i+chunk, len(ids))}/{len(ids)} into ChromaDB")

    final_count = collection.count()
    logger.info(f"\n✅ Done! ChromaDB now has {final_count} movies.\n")


if __name__ == "__main__":
    main()
