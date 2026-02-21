import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import requests

logger = logging.getLogger(__name__)

class WebPersonaService:
    def __init__(self):
        self._roots_dir = Path(__file__).parent.parent.parent
        self.data_file = self._roots_dir / "data" / "personas" / "nuanced_reddit_comments.csv"
        self.model = None
        self.centroids = {}
        self.cluster_meta = {}
        self._initialized = False

    def initialize(self):
        """Lazy load the sentence transformer and centroids to save memory at boot."""
        if self._initialized: return
        
        from sentence_transformers import SentenceTransformer
        logger.info("Initializing Web Persona Service & Sentence Transformer...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        
        if not self.data_file.exists():
            logger.warning(f"Persona dataset not found at {self.data_file}")
            return
            
        df = pd.read_csv(self.data_file)
        
        # Calculate centroids and extract demographic meta per group
        unique_groups = df["cluster_name"].unique()
        for group in unique_groups:
            if group == "Noise": continue
            
            group_df = df[df["cluster_name"] == group]
            texts = group_df["comment_text"].tolist()
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            self.centroids[group] = np.mean(embeddings, axis=0)
            
            # Use TF-IDF to snag top keywords representing this group's conversation
            vectorizer = TfidfVectorizer(stop_words='english', max_df=0.9, min_df=2)
            try:
                tfidf_matrix = vectorizer.fit_transform(texts)
                sum_scores = tfidf_matrix.sum(axis=0)
                words = vectorizer.get_feature_names_out()
                word_scores = [(words[i], sum_scores[0, i]) for i in range(len(words))]
                word_scores.sort(key=lambda x: x[1], reverse=True)
                top_keywords = [w[0] for w in word_scores[:5]]
            except Exception:
                top_keywords = ["general"]
                
            # Extract distinct demographics (we generated uniform ones per group)
            self.cluster_meta[group] = {
                "keywords": top_keywords,
                "age_demographic": group_df["age_demographic"].iloc[0],
                "gender_demographic": group_df["gender_demographic"].iloc[0],
                "region": group_df["region"].iloc[0],
                "size": len(group_df)
            }
            
        self._initialized = True
        logger.info("Web Persona Service fully initialized.")

    def calculate_affinity(self, fingerprint: dict) -> list[dict]:
        """Compare a movie fingerprint against the pre-calculated nuance personas."""
        self.initialize()
        if not self.centroids:
            return [{"error": "Persona Dataset Missing"}]
            
        film_analysis = fingerprint.get("film_analysis", {})
        top_kws = fingerprint.get("top_keywords", [])
        
        # Build synthetic context string representing the trailer
        trailer_text = f"{' '.join(film_analysis.get('genre_hints', []))}. {' '.join(top_kws)}. {film_analysis.get('description', '')}"
        trailer_vec = self.model.encode([trailer_text], convert_to_numpy=True)
        
        results = []
        for group, centroid in self.centroids.items():
            sim = cosine_similarity(trailer_vec, centroid.reshape(1, -1))[0][0]
            
            meta = self.cluster_meta.get(group, {})
            keywords = meta.get("keywords", [])
            
            # Generate Subreddit Marketing Suggestions via Local Ollama
            subreddits = "r/movies, r/entertainment"
            try:
                prompt = (
                    f"You are a film marketing expert. I have an audience segment that "
                    f"talks about these topics: {', '.join(keywords)}. "
                    f"Provide EXACTLY THREE highly relevant subreddits where I should advertise a movie to them. "
                    f"Return ONLY the subreddit names separated by commas like 'r/scifi, r/VFX, r/action'. "
                    f"Do not write any introductory text."
                )
                
                response = requests.post(
                    "http://127.0.0.1:11434/api/generate",
                    json={
                        "model": "llama3:latest",
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.4}
                    },
                    timeout=10 # dont hang the web server
                )
                if response.status_code == 200:
                    subreddits = response.json().get("response", subreddits).strip()
            except Exception as e:
                logger.error(f"Failed to fetch subreddits from Ollama: {e}")

            results.append({
                "persona": group,
                "affinity_score": float(sim),
                "top_keywords": keywords,
                "size": meta.get("size", 0),
                "demographics": {
                    "subreddits": subreddits
                }
            })
            
        # Rank by highest matching demographic
        results.sort(key=lambda x: x["affinity_score"], reverse=True)
        return results
