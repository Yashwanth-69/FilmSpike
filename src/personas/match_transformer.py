import os
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = _ROOT / "data" / "personas"
FINGERPRINT_FILE = _ROOT / "data" / "output" / "test.json"
CLUSTERED_DATA_FILE = OUTPUT_DIR / "clustered_comments.csv"
FINAL_OUTPUT_FILE = OUTPUT_DIR / "transformer_persona_affinity.json"

def extract_top_keywords(texts: list[str], top_n: int = 5) -> str:
    """Uses TF-IDF to find a descriptive label for the cluster."""
    if len(texts) < 2:
        return "Unknown Persona"
    try:
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_df=0.9, min_df=2)
        tfidf_matrix = vectorizer.fit_transform(texts)
        sum_scores = tfidf_matrix.sum(axis=0)
        
        words = vectorizer.get_feature_names_out()
        word_scores = [(words[i], sum_scores[0, i]) for i in range(len(words))]
        word_scores.sort(key=lambda x: x[1], reverse=True)
        return ", ".join([w[0] for w in word_scores[:top_n]])
    except ValueError:
        return "General"

def generate_trailer_text(fingerprint: dict) -> str:
    """Extract descriptive text from the fingerprint to embed for Persona matching."""
    top_keywords = fingerprint.get("top_keywords", [])
    film_analysis = fingerprint.get("film_analysis", {})
    
    genres = film_analysis.get("genre_hints", [])
    synopsis = film_analysis.get("description", "")
    
    # We want a paragraph that represents the trailer's thematic core
    # "A [genres] trailer focusing on [keywords]. Description: [synopsis]"
    
    context = []
    if genres:
        context.append(f"A {' and '.join(genres)} film or trailer.")
    if top_keywords:
        context.append(f"It features themes like {', '.join(top_keywords)}.")
    if synopsis:
        context.append(f"Synopsis: {synopsis}")
        
    final_text = " ".join(context)
    logger.info(f"Generated Trailer Context for Embedding: \n'{final_text}'")
    return final_text

def main():
    if not FINGERPRINT_FILE.exists():
        logger.error(f"Cannot find fingerprint at {FINGERPRINT_FILE}")
        return
    if not CLUSTERED_DATA_FILE.exists():
        logger.error(f"Cannot find clustered data at {CLUSTERED_DATA_FILE}")
        return

    logger.info("Loading Transformer fingerprint...")
    with open(FINGERPRINT_FILE, "r") as f:
        fingerprint = json.load(f)
        
    trailer_context = generate_trailer_text(fingerprint)
    
    logger.info("Loading SentenceTransformer model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    logger.info(f"Loading HDBSCAN persona clusters from {CLUSTERED_DATA_FILE}...")
    df = pd.read_csv(CLUSTERED_DATA_FILE)
    
    clusters = df["hdbscan_cluster_id"].unique()
    valid_clusters = [c for c in clusters if c != -1]
    
    # 1. Embed the trailer text
    trailer_vector = model.encode([trailer_context], convert_to_numpy=True)
    
    results = []
    
    for c in valid_clusters:
        cluster_df = df[df["hdbscan_cluster_id"] == c]
        texts = cluster_df["comment_text"].tolist()
        
        # 2. Extract Persona Name (Keywords)
        persona_name = f"Persona Cluster {c}: [{extract_top_keywords(texts)}]"
        
        # 3. Calculate Centroid
        embeddings = model.encode(texts, convert_to_numpy=True)
        centroid = np.mean(embeddings, axis=0)
        
        # 4. Compute Affinity
        sim = cosine_similarity(trailer_vector, centroid.reshape(1, -1))[0][0]
        
        results.append({
            "cluster_id": int(c), # JSON serializable
            "persona_summary": persona_name,
            "comment_count": len(texts),
            "affinity_score": float(sim)
        })
        
    # Rank by affinity
    results.sort(key=lambda x: x["affinity_score"], reverse=True)
    
    # Display and Save
    logger.info("\n--- Target Audience Affinity for 'sample.mp4' (Transformers) ---")
    for r in results:
        logger.info(f"Rank {results.index(r)+1}: {r['persona_summary']} (Score: {r['affinity_score']:.4f})")
    
    output_data = {
        "movie_source": "sample.mp4 (Transformers)",
        "generated_trailer_context": trailer_context,
        "matched_personas": results
    }
    
    with open(FINAL_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)
        
    logger.info(f"\nSaved final output to {FINAL_OUTPUT_FILE}")

if __name__ == "__main__":
    main()
