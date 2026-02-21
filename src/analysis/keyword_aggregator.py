"""
keyword_aggregator.py
Phase 2 — Aggregate per-shot zero-shot concepts into film-level keywords
and compute cinematic feature summary (pacing, color mood, audio mood).
"""
import json
import logging
import numpy as np
from typing import Dict, Any, List
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


class KeywordAggregator:
    """
    Reads a Phase-1 fingerprint and produces:
      - top_keywords: ranked list of concept strings
      - film_analysis: pacing, color_mood, audio_mood, dominant_emotion, genre_hints
    """

    # Map concept groups → genres, with priority weight
    # Higher weight = stronger genre signal when this group is dominant
    GENRE_HINT_MAP = {
        "sci_fi":    [("Science Fiction", 3.0), ("Action", 1.5)],
        "action":    [("Action", 3.0), ("Thriller", 1.5)],
        "nature":    [("Adventure", 2.0), ("Drama", 1.0)],
        "urban":     [("Crime", 2.0), ("Thriller", 1.5), ("Drama", 1.0)],
        "people":    [("Drama", 1.5), ("Romance", 1.0)],
        "emotional": [("Drama", 1.0), ("Horror", 0.5)],  # Lowered weight — emotions alone don't define genre
    }

    EMOTION_GENRE_MAP = {
        "fear":      [("Horror", 1.5), ("Thriller", 1.0)],
        "happiness": [("Comedy", 1.5), ("Romance", 1.0), ("Animation", 0.5)],
        "anger":     [("Action", 1.5), ("Crime", 1.0)],
        "sadness":   [("Drama", 1.5)],
        "surprise":  [("Thriller", 1.0), ("Mystery", 0.5)],
    }

    def aggregate(self, fingerprint: Dict[str, Any]) -> Dict[str, Any]:
        """
        Args:
            fingerprint: the full JSON dict from FilmFingerprinter.fingerprint()

        Returns:
            dict with keys: top_keywords, film_analysis
        """
        shots: List[Dict] = fingerprint.get("shots", [])
        if not shots:
            logger.warning("No shots found in fingerprint — returning empty analysis")
            return {"top_keywords": [], "film_analysis": {}}

        # ── 1. Aggregate zero-shot concepts ──────────────────────────────────
        concept_weights: Dict[str, float] = defaultdict(float)
        total_duration: float = sum(s.get("duration", 1.0) for s in shots)

        for shot in shots:
            shot_dur = shot.get("duration", 1.0)
            weight_factor = shot_dur / total_duration  # longer shots count more

            zs = shot.get("zero_shot_concepts", {})
            for entry in zs.get("primary_concepts", []):
                concept = entry.get("concept", "")
                conf = entry.get("confidence", 0.0)
                concept_weights[concept] += conf * weight_factor

            # Also pull from detected_concepts for completeness
            for concept, data in zs.get("detected_concepts", {}).items():
                conf = data.get("avg_confidence", 0.0)
                concept_weights[concept] += conf * weight_factor * 0.5  # half weight secondary

        # Sort by weight, take top-15
        sorted_concepts = sorted(concept_weights.items(), key=lambda x: x[1], reverse=True)
        top_keywords = [c for c, _ in sorted_concepts[:15] if c]

        # ── 2. Pacing analysis (from shot durations) ─────────────────────────
        durations = [s.get("duration", 1.0) for s in shots]
        avg_shot_dur = float(np.mean(durations))
        if avg_shot_dur < 2.0:
            pacing = "fast"
        elif avg_shot_dur < 5.0:
            pacing = "medium"
        else:
            pacing = "slow"

        # ── 3. Color mood (from brightness across shots) ──────────────────────
        brightness_vals = []
        for shot in shots:
            vf = shot.get("visual_features", {})
            bval = vf.get("brightness_mean", None)
            if bval is not None:
                brightness_vals.append(bval)

        if brightness_vals:
            avg_brightness = float(np.mean(brightness_vals))
            if avg_brightness < 60:
                color_mood = "dark"
            elif avg_brightness < 140:
                color_mood = "moody"
            else:
                color_mood = "bright"
        else:
            color_mood = "unknown"

        # ── 4. Dominant emotion ───────────────────────────────────────────────
        summary = fingerprint.get("summary", {})
        dominant_emotions: Dict[str, int] = summary.get("dominant_emotions", {})
        dominant_emotion = max(dominant_emotions, key=dominant_emotions.get) if dominant_emotions else "neutral"

        # ── 5. Audio mood (from librosa features) ─────────────────────────────
        tempo_vals = []
        silence_ratios = []
        for shot in shots:
            af = shot.get("audio_features", {})
            inner = af.get("audio_features", {})
            if inner:
                t = inner.get("tempo", 0.0)
                if t > 0:
                    tempo_vals.append(t)
                silence_ratios.append(inner.get("silence_ratio", 1.0))

        avg_tempo = float(np.mean(tempo_vals)) if tempo_vals else 0.0
        avg_silence = float(np.mean(silence_ratios)) if silence_ratios else 1.0

        if avg_silence > 0.7:
            audio_mood = "silent/ambient"
        elif avg_tempo > 140:
            audio_mood = "intense/action"
        elif avg_tempo > 100:
            audio_mood = "energetic"
        else:
            audio_mood = "calm/dramatic"

        # ── 6. Genre hints (confidence-weighted) ─────────────────────────────
        # Use concept_groups dict: {'sci_fi': ['laser', 'robot'], 'action': [...]}
        # Weight each group by sum of confidences of its member concepts
        group_scores: Dict[str, float] = defaultdict(float)
        for shot in shots:
            shot_dur = shot.get("duration", 1.0)
            weight_factor = shot_dur / total_duration
            zs = shot.get("zero_shot_concepts", {})
            concept_groups = zs.get("concept_groups", {})
            detected = zs.get("detected_concepts", {})

            for group, concepts in concept_groups.items():
                for concept in concepts:
                    conf = detected.get(concept, {}).get("avg_confidence", 0.25)
                    group_scores[group] += conf * weight_factor

        # Map groups → genres using confidence-weighted scores
        genre_weights: Dict[str, float] = defaultdict(float)
        for group, score in group_scores.items():
            for genre, weight in self.GENRE_HINT_MAP.get(group, []):
                genre_weights[genre] += score * weight

        # Emotion contribution — only dominant emotion gets its genres added
        dom_emo_genre_pairs = self.EMOTION_GENRE_MAP.get(dominant_emotion, [])
        for genre, weight in dom_emo_genre_pairs:
            genre_weights[genre] += weight

        # Take top genres by accumulated weight
        top_genre_hints = [g for g, _ in sorted(genre_weights.items(), key=lambda x: x[1], reverse=True)[:5]]

        # ── 7. Mean CLIP embedding (film-level) ────────────────────────────────
        embeddings = []
        weights = []
        for shot in shots:
            emb = shot.get("visual_features", {}).get("clip_embedding")
            if emb:
                embeddings.append(np.array(emb, dtype=np.float32))
                weights.append(shot.get("duration", 1.0))

        if embeddings:
            weights_arr = np.array(weights, dtype=np.float32)
            weights_arr /= weights_arr.sum()
            film_vector = np.average(embeddings, axis=0, weights=weights_arr)
            film_vector = (film_vector / np.linalg.norm(film_vector)).tolist()
        else:
            film_vector = None

        return {
            "top_keywords": top_keywords,
            "film_vector": film_vector,       # 512-dim, used by MovieMatcher internally
            "film_analysis": {
                "pacing": pacing,
                "avg_shot_duration_sec": round(avg_shot_dur, 2),
                "color_mood": color_mood,
                "dominant_emotion": dominant_emotion,
                "audio_mood": audio_mood,
                "genre_hints": top_genre_hints,
                "shot_count": len(shots),
                "total_duration_sec": round(total_duration, 2),
            }
        }
