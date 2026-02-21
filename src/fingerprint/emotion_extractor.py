import json
import torch
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Any
from transformers import pipeline
import cv2
from PIL import Image

# Load concept config once at module level
_CONFIG_PATH = Path(__file__).parent / "config" / "concepts.json"
with _CONFIG_PATH.open("r", encoding="utf-8") as _f:
    _CONCEPTS = json.load(_f)

logger = logging.getLogger(__name__)

class EmotionExtractor:
    """Analyzes emotional content from faces using a transformer model."""
    
    def __init__(self, device: str = None):
        """
        Initialize the emotion classifier.
        
        Args:
            device: 'cpu' or 'cuda'. If None, auto-select.
        """
        if device is None:
            device = 0 if torch.cuda.is_available() else -1
        else:
            device = 0 if device == 'cuda' else -1
        
        logger.info(f"Loading emotion classification model on device {device}...")
        self.classifier = pipeline(
            "image-classification",
            model="dima806/facial_emotions_image_detection",
            device=device
        )
        
        # Load emotion categories from external config
        self.emotion_categories = _CONCEPTS["emotion"]["categories"]
        logger.info("Emotion model loaded.")
    
    def analyze_frame_emotion(self, frame: np.ndarray) -> Dict[str, float]:
        """
        Analyze emotions in a single frame.
        
        Args:
            frame: numpy array in BGR format (OpenCV default)
            
        Returns:
            Dictionary mapping emotion -> confidence.
        """
        if frame is None or frame.size == 0:
            return {cat: 0.0 for cat in self.emotion_categories}
        
        # Ensure frame is uint8
        if frame.dtype != np.uint8:
            logger.debug(f"Converting frame dtype from {frame.dtype} to uint8")
            if frame.dtype.kind == 'f':
                frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert to PIL Image (required by the pipeline)
        pil_image = Image.fromarray(frame_rgb)
        
        try:
            results = self.classifier(pil_image)
            emotions = {r['label'].lower(): r['score'] for r in results}
            for cat in self.emotion_categories:
                emotions.setdefault(cat, 0.0)
            return emotions
        except Exception as e:
            logger.warning(f"Emotion analysis failed: {e}")
            return {cat: 0.0 for cat in self.emotion_categories}
    
    def analyze_shot_emotions(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """
        Analyze emotional arc across a shot by processing a subset of frames.
        
        Args:
            frames: List of frames from the shot.
            
        Returns:
            Aggregated emotion information.
        """
        if not frames:
            return {}
        
        # To save time, sample at most 5 frames per shot (evenly spaced)
        sample_rate = max(1, len(frames) // 5)
        sampled_frames = frames[::sample_rate][:5]  # at most 5
        
        emotions_list = [self.analyze_frame_emotion(f) for f in sampled_frames]
        
        # Calculate mean emotions across sampled frames
        mean_emotions = {}
        for cat in self.emotion_categories:
            values = [e.get(cat, 0) for e in emotions_list]
            mean_emotions[cat] = float(np.mean(values))
        
        # Find dominant emotion (highest mean)
        dominant_emotion = max(mean_emotions, key=mean_emotions.get)
        
        # Calculate emotional variance (indicates change within shot)
        emotion_variance = {}
        for cat in self.emotion_categories:
            values = [e.get(cat, 0) for e in emotions_list]
            emotion_variance[cat] = float(np.var(values))
        
        # Sequence of dominant emotions per sampled frame (for arc)
        emotion_sequence = [max(e, key=e.get) for e in emotions_list]
        
        return {
            "mean_emotions": mean_emotions,
            "dominant_emotion": dominant_emotion,
            "dominant_confidence": mean_emotions[dominant_emotion],
            "emotion_variance": emotion_variance,
            "emotional_volatility": float(np.mean(list(emotion_variance.values()))),
            "emotion_sequence": emotion_sequence,
            "frames_analyzed": len(sampled_frames)
        }