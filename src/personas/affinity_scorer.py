import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = _ROOT / "data" / "personas"
INPUT_FILE = DATA_DIR / "clustered_comments.csv"

# Pre-defined trailer concepts for our simulation
TARGET_TRAILERS = {
    # Expected highest affinity: Analytical Sci-Fi Persona
    "Sci-Fi Trailer": "A complex, slow-burn teaser showing intricate machinery, space physics, and deep philosophical dialogue about the universe and alien lore.",
    # Expected highest affinity: Casual Blockbuster Persona
    "Action Blockbuster Trailer": "A high-octane montage of massive explosions, giant robots punching each other, fast cars, and a hyped up hip-hop soundtrack.",
    # Expected highest affinity: Character Drama Persona
    "Romance Drama Trailer": "A heartfelt, emotional trailer focusing on two characters holding hands, crying, and overcoming a tragic relationship obstacle together."
}

def load_cluster_centroids(df: pd.DataFrame, model: SentenceTransformer) -> dict:
    """Calculates the centroid embedding for each discovered cluster."""
    clusters = df["hdbscan_cluster_id"].unique()
    clusters = [c for c in clusters if c != -1] # Ignore noise
    
    centroids = {}
    for c in clusters:
        cluster_texts = df[df["hdbscan_cluster_id"] == c]["comment_text"].tolist()
        
        # In a strict pipeline, we'd load the pre-computed embeddings matrix from clustering.py
        # For this simulation script, we just re-embed the cluster's text to find its center
        embeddings = model.encode(cluster_texts, convert_to_numpy=True)
        
        # Calculate geometric center of the cluster
        centroid = np.mean(embeddings, axis=0)
        centroids[c] = centroid
        
    return centroids

def main():
    if not INPUT_FILE.exists():
        logger.error("Missing clustered dataset. Run clustering.py first.")
        return
        
    df = pd.read_csv(INPUT_FILE)
    
    logger.info("Loading SentenceTransformer model to compute centroids and trailer vectors...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    logger.info("Computing cluster centroids...")
    centroids = load_cluster_centroids(df, model)
    
    logger.info("\n--- Affinity Ranking Simulation ---\n")
    
    # Evaluate every simulated "trailer" against every extracted "Persona"
    for trailer_name, trailer_desc in TARGET_TRAILERS.items():
        logger.info(f"Target Trailer: {trailer_name}")
        logger.info(f"Content: '{trailer_desc}'")
        
        # Our simulated "feature extraction" via sentence embeddings
        trailer_vec = model.encode([trailer_desc], convert_to_numpy=True)
        
        scores = []
        for cluster_id, centroid_vec in centroids.items():
            # True ground label to help us interpret the cluster visually
            mode_label = df[df["hdbscan_cluster_id"] == cluster_id]["true_cluster_id"].mode()[0]
            
            # Match 1:N (1 trailer vector against N cluster centroids)
            sim = cosine_similarity(trailer_vec, centroid_vec.reshape(1, -1))[0][0]
            scores.append({
                "cluster_vid": cluster_id,
                "true_theme": mode_label,
                "affinity": sim
            })
            
        # Rank the clusters by how much they engage with this trailer
        scores.sort(key=lambda x: x["affinity"], reverse=True)
        
        for rank, s in enumerate(scores, 1):
            logger.info(f"  Rank {rank}: Cluster {s['cluster_vid']} (Theme Group: {s['true_theme']}) -> Score: {s['affinity']:.4f}")
        logger.info("-" * 50)

if __name__ == "__main__":
    main()
