import pandas as pd
import numpy as np
import logging
import argparse
from pathlib import Path
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
from sklearn.metrics import adjusted_rand_score, v_measure_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = _ROOT / "data" / "personas"
INPUT_FILE = DATA_DIR / "synthetic_reddit_comments.csv"
OUTPUT_FILE = DATA_DIR / "clustered_comments.csv"

def embed_comments(df: pd.DataFrame, text_col: str = "comment_text", model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Embed text comments using sentence-transformers."""
    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    
    texts = df[text_col].tolist()
    logger.info(f"Embedding {len(texts)} comments...")
    
    # Encode with progress bar
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings

def cluster_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Run UMAP for dimensionality reduction followed by HDBSCAN for clustering."""
    
    logger.info("Running UMAP dimensionality reduction (512 -> 5D)...")
    # Project down to 5 dimensions to help HDBSCAN handle the "curse of dimensionality"
    reducer = umap.UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric='cosine',
        random_state=42
    )
    reduced_embeddings = reducer.fit_transform(embeddings)
    
    logger.info("Running HDBSCAN clustering...")
    # HDBSCAN parameters tuned for small-ish textual datasets (min cluster size 10)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=10,
        min_samples=5,
        metric='euclidean',
        cluster_selection_method='eom'
    )
    cluster_labels = clusterer.fit_predict(reduced_embeddings)
    
    num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    num_noise = list(cluster_labels).count(-1)
    
    logger.info(f"HDBSCAN discovered {num_clusters} clusters. Identified {num_noise} noise points (-1).")
    return cluster_labels

def evaluate_clustering(true_labels: pd.Series, predicted_labels: np.ndarray):
    """Calculate clustering metrics against our synthetic ground truth."""
    logger.info("Evaluating clustering performance against ground truth...")
    
    # Note: HDBSCAN assigns -1 to noise. 
    # Adjusted Rand Index (ARI): Similarity measure between two clusterings.
    # V-Measure: Harmonic mean of homogeneity and completeness.
    
    ari = adjusted_rand_score(true_labels, predicted_labels)
    v_measure = v_measure_score(true_labels, predicted_labels)
    
    logger.info(f"Adjusted Rand Index (ARI): {ari:.4f} (1.0 is perfect match, 0.0 is random)")
    logger.info(f"V-Measure: {v_measure:.4f} (1.0 is perfect match)")
    
    # Print cluster distribution mapping
    df_eval = pd.DataFrame({"True": true_labels, "Pred": predicted_labels})
    crosstab = pd.crosstab(df_eval["True"], df_eval["Pred"], margins=True)
    logger.info(f"\nCluster Mapping Crosstab (Columns = Predicted HDBSCAN ID, Rows = True Synthetic ID):\n{crosstab}\n")
    
def main():
    if not INPUT_FILE.exists():
        logger.error(f"Input file not found: {INPUT_FILE}")
        logger.error("Please run synthetic_data_gen.py first.")
        return
        
    logger.info(f"Processing dataset: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    
    if len(df) == 0:
        logger.error("Dataset is empty.")
        return
        
    embeddings = embed_comments(df)
    
    cluster_labels = cluster_embeddings(embeddings)
    
    df["hdbscan_cluster_id"] = cluster_labels
    
    # Validate against our engineered ground truth
    if "true_cluster_id" in df.columns:
        evaluate_clustering(df["true_cluster_id"], cluster_labels)
        
    df.to_csv(OUTPUT_FILE, index=False)
    logger.info(f"Saved clustered dataset with {len(df)} records to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
