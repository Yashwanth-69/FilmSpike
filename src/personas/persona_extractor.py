import pandas as pd
import logging
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = _ROOT / "data" / "personas"
INPUT_FILE = DATA_DIR / "clustered_comments.csv"

def extract_top_keywords(texts: list[str], top_n: int = 10) -> list[str]:
    """Uses TF-IDF to find the most distinguishing keywords/phrases in a set of texts."""
    if not texts:
        return []
        
    try:
        # Stop_words roughly filter common english, ngram_range gets phrases like "giant robot"
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_df=0.9, min_df=2)
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # Sum tfidf scores across all documents in this cluster
        sum_scores = tfidf_matrix.sum(axis=0)
        
        words = vectorizer.get_feature_names_out()
        word_scores = [(words[i], sum_scores[0, i]) for i in range(len(words))]
        
        # Sort by score descending
        word_scores.sort(key=lambda x: x[1], reverse=True)
        return [word for word, score in word_scores[:top_n]]
    except ValueError:
        return ["Not enough data"]

def main():
    if not INPUT_FILE.exists():
        logger.error(f"Missing {INPUT_FILE}. Run clustering.py first.")
        return
        
    df = pd.read_csv(INPUT_FILE)
    
    unique_clusters = df["hdbscan_cluster_id"].unique()
    unique_clusters = sorted([c for c in unique_clusters if c != -1]) # ignore noise (-1)
    
    logger.info("Extracting Persona Keywords for discovered clusters...")
    
    for cluster_id in unique_clusters:
        cluster_df = df[df["hdbscan_cluster_id"] == cluster_id]
        
        # Determine dominant ground-truth label to see if we recovered the right structure
        true_label = cluster_df["true_cluster_id"].mode()[0]
        
        texts = cluster_df["comment_text"].tolist()
        keywords = extract_top_keywords(texts)
        
        logger.info(f"\n--- Discovered Persona: Cluster {cluster_id} ---")
        logger.info(f"Size: {len(cluster_df)} comments")
        logger.info(f"Dominant True Label: {true_label}")
        logger.info(f"Top Keywords: {', '.join(keywords)}")
        
        # In a real pipeline, we would send these top keywords + a random sample of comments to an LLM
        # to generate a natural language "Persona Description" (e.g., "The Technical Sci-Fi Buff").
        # Here, we validate that the TF-IDF keywords clearly map back to the synthetic prompts.

if __name__ == "__main__":
    main()
