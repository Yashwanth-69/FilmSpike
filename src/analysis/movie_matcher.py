"""
movie_matcher.py
Phase 2 — Query ChromaDB with the film's CLIP image vector to find similar movies.
On first run (or if ChromaDB is empty), auto-ingests from movies_fallback.csv.
"""
import json
import logging
import numpy as np
import csv
from pathlib import Path
from typing import Dict, Any, List, Optional

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

# Paths
_ROOT = Path(__file__).parent.parent.parent   # project root
_DB_PATH = _ROOT / "data" / "chroma_db"
_FALLBACK_CSV = _ROOT / "data" / "movies_fallback.csv"
_COLLECTION_NAME = "movies"
_MIN_MOVIES = 200  # auto-ingest fallback CSV only if collection is nearly empty


class MovieMatcher:
    """
    Queries a ChromaDB collection of CLIP-encoded movie descriptions.
    Uses the film's duration-weighted mean CLIP image embedding as query vector.
    """

    def __init__(self):
        _DB_PATH.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(_DB_PATH))
        self.collection = self.client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

        # Auto-ingest bundled CSV if DB is sparse
        count = self.collection.count()
        logger.info(f"ChromaDB has {count} movies")
        if count < _MIN_MOVIES:
            logger.info("Collection sparse — auto-ingesting from fallback CSV...")
            self._ingest_csv(_FALLBACK_CSV)

    def _ingest_csv(self, csv_path: Path):
        """Ingest movies_fallback.csv into ChromaDB using pre-stored embeddings."""
        if not csv_path.exists():
            logger.warning(f"Fallback CSV not found at {csv_path}. Skipping auto-ingest.")
            return

        import torch
        import clip as clip_module

        device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        logger.info(f"Loading CLIP for movie text encoding on {device}...")
        model, _ = clip_module.load("ViT-B/32", device=device)
        model.eval()

        ids, embeddings, documents, metadatas = [], [], [], []

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Deduplicate by tmdb_id
        seen_ids = set()
        deduped = []
        for row in rows:
            rid = str(row.get("tmdb_id", "")).strip()
            if rid and rid not in seen_ids:
                seen_ids.add(rid)
                deduped.append(row)
        rows = deduped

        logger.info(f"Encoding {len(rows)} movies from CSV...")
        batch_size = 32

        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            texts = []
            for row in batch:
                genres = row.get("genres", "")
                overview = row.get("overview", "")
                text = f"{genres}: {overview}".strip()
                if not text:
                    text = row.get("title", "unknown movie")
                texts.append(text)

            # Encode with CLIP text encoder
            import torch
            tokens = clip_module.tokenize(texts, truncate=True).to(device)
            with torch.no_grad():
                feats = model.encode_text(tokens)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                feats_np = feats.cpu().numpy()

            for j, row in enumerate(batch):
                movie_id = f"tmdb_{row.get('tmdb_id', i+j)}"
                ids.append(movie_id)
                embeddings.append(feats_np[j].tolist())
                documents.append(texts[j])
                metadatas.append({
                    "title":        row.get("title", ""),
                    "year":         int(row.get("year", 0)) if row.get("year") else 0,
                    "genres":       row.get("genres", ""),
                    "revenue":      int(row.get("revenue", 0)) if row.get("revenue") else 0,
                    "release_date": row.get("release_date", ""),
                    "vote_average": float(row.get("vote_average", 0)) if row.get("vote_average") else 0.0,
                    "tmdb_id":      int(row.get("tmdb_id", 0)) if row.get("tmdb_id") else 0,
                })

        if ids:
            # Upsert in batches to avoid ChromaDB limits
            batch_size = 100
            for i in range(0, len(ids), batch_size):
                self.collection.upsert(
                    ids=ids[i:i+batch_size],
                    embeddings=embeddings[i:i+batch_size],
                    documents=documents[i:i+batch_size],
                    metadatas=metadatas[i:i+batch_size],
                )
            logger.info(f"Ingested {len(ids)} movies into ChromaDB")

    def find_similar(
        self,
        fingerprint: Dict[str, Any],
        film_vector: List[float],
        top_k: int = 10,
        genre_hints: Optional[List[str]] = None,
        top_keywords: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Dual-query ChromaDB:
          Query 1 — CLIP image vector (visual aesthetics match)
          Query 2 — CLIP text encoding of film's top_keywords (semantic concept match)
          Final score = 0.6 × visual_sim + 0.4 × keyword_sim

        Args:
            fingerprint: full fingerprint dict
            film_vector: 512-dim CLIP image embedding (duration-weighted mean)
            top_k: number of results to return
            genre_hints: genre strings for re-ranking boost
            top_keywords: zero-shot concept keywords from KeywordAggregator
        """
        count = self.collection.count()
        if count == 0:
            logger.warning("ChromaDB is empty — no similar movies found")
            return []

        n_candidates = min(50, count)  # wider net for blended re-ranking

        # ── Query 1: CLIP image vector ────────────────────────────────────────
        img_results = self.collection.query(
            query_embeddings=[film_vector],
            n_results=n_candidates,
            include=["metadatas"],
        )
        img_ids = [f"tmdb_{m.get('tmdb_id', 0)}" for m in img_results["metadatas"][0]]

        # ── Query 2: CLIP text encoding of film's own keywords ────────────────
        kw_ids = []
        kw_vec_np = None
        if top_keywords:
            try:
                import torch
                import clip as clip_module

                device = "cuda" if torch.cuda.is_available() else "cpu"
                # Lazy-load CLIP only once per matcher instance
                if not hasattr(self, "_clip_model"):
                    logger.info("Loading CLIP text encoder for keyword query...")
                    self._clip_model, _ = clip_module.load("ViT-B/32", device=device)
                    self._clip_model.eval()
                    self._clip_device = device

                # Build keyword query string — join top keywords
                kw_text = ", ".join(top_keywords[:10])
                tokens = clip_module.tokenize([kw_text], truncate=True).to(self._clip_device)
                with torch.no_grad():
                    kw_vec = self._clip_model.encode_text(tokens)
                    kw_vec = kw_vec / kw_vec.norm(dim=-1, keepdim=True)
                    kw_vec_np = kw_vec.cpu().numpy()[0]
                    kw_vec_list = kw_vec_np.tolist()

                kw_results = self.collection.query(
                    query_embeddings=[kw_vec_list],
                    n_results=n_candidates,
                    include=["metadatas"],
                )
                kw_ids = [f"tmdb_{m.get('tmdb_id', 0)}" for m in kw_results["metadatas"][0]]

            except Exception as e:
                logger.warning(f"Keyword query failed: {e} — using image query only")

        all_ids = list(set(img_ids) | set(kw_ids))
        if "tmdb_68726" in img_ids:
            print("DEBUG: Pacific Rim IS in img_ids!")
        else:
            print("DEBUG: Pacific Rim is NOT in img_ids!")
        
        if "tmdb_68726" in kw_ids:
            print("DEBUG: Pacific Rim IS in kw_ids!")
        else:
            print("DEBUG: Pacific Rim is NOT in kw_ids!")

        if not all_ids:
            return []

        # ── Fetch embeddings for all candidate IDs to compute exact similarities ──
        full_data = self.collection.get(
            ids=all_ids,
            include=["metadatas", "embeddings"]
        )

        film_vec_np = np.array(film_vector)
        norm_film = np.linalg.norm(film_vec_np)
        norm_kw = np.linalg.norm(kw_vec_np) if kw_vec_np is not None else 0.0

        # ── Blend scores & build result list ──────────────────────────────────
        movies = []
        for meta, emb in zip(full_data["metadatas"], full_data["embeddings"]):
            if meta is None or emb is None:
                continue

            emb_np = np.array(emb)
            norm_emb = np.linalg.norm(emb_np)

            v_sim = 0.0
            if norm_emb > 0 and norm_film > 0:
                cos_sim_v = float(np.dot(emb_np, film_vec_np) / (norm_emb * norm_film))
                v_sim = (cos_sim_v + 1.0) / 2.0

            k_sim = 0.0
            if kw_vec_np is not None and norm_emb > 0 and norm_kw > 0:
                cos_sim_k = float(np.dot(emb_np, kw_vec_np) / (norm_emb * norm_kw))
                k_sim = (cos_sim_k + 1.0) / 2.0

            if kw_vec_np is not None:
                # Both queries ran — blend 60/40
                blended = 0.6 * v_sim + 0.4 * k_sim
            else:
                # Keyword query didn't run — use image score only
                blended = v_sim

            # Explicit Zero-Shot Keyword Match Boost
            # Rewards exact metadata overlap (e.g., "giant robot")
            if top_keywords:
                movie_keywords = [k.strip().lower() for k in meta.get("keywords", "").split(",")]
                for tk in top_keywords:
                    tk_lower = tk.lower()
                    if tk_lower and any(tk_lower in mk or mk in tk_lower for mk in movie_keywords if mk):
                        blended = min(1.0, blended + 0.05)

            # Genre boost: +0.03 per matching genre
            if genre_hints:
                movie_genres = meta.get("genres", "")
                for hint in genre_hints:
                    if hint.lower() in movie_genres.lower():
                        blended = min(1.0, blended + 0.03)

            # Soft recency penalty — very old films get a score discount
            year = meta.get("year", 2000)
            if year and year < 1995:
                age_penalty = min(0.12, (1995 - year) * 0.006)  # max -0.12 for films ≥20yrs before 1995
                blended = max(0.0, blended - age_penalty)

            # Filter out low-confidence matches (unlikely to be relevant)
            if blended < 0.70:
                continue

            movies.append({
                "title":            meta.get("title", ""),
                "year":             meta.get("year", 0),
                "genres":           [g.strip() for g in meta.get("genres", "").split(",") if g.strip()],
                "revenue":          meta.get("revenue", 0),
                "release_date":     meta.get("release_date", ""),
                "vote_average":     meta.get("vote_average", 0.0),
                "similarity_score": round(blended, 4),
                "_visual_score":    round(v_sim, 4),
                "_keyword_score":   round(k_sim, 4),
            })
        movies.sort(key=lambda x: x["similarity_score"], reverse=True)
        return movies[:top_k]


