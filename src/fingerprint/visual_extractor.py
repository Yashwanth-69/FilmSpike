import json
import torch
import clip
from pathlib import Path
from PIL import Image
import numpy as np
from typing import List, Dict, Any
import logging
import cv2
from sklearn.cluster import KMeans

# Load concept config once at module level
_CONFIG_PATH = Path(__file__).parent / "config" / "concepts.json"
with _CONFIG_PATH.open("r", encoding="utf-8") as _f:
    _CONCEPTS = json.load(_f)

logger = logging.getLogger(__name__)

class VisualExtractor:
    """Extracts visual features using CLIP multimodal model"""
    
    def __init__(self, model_name: str = "ViT-B/32", device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Loading CLIP model on {self.device}")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        
        # Load categories from external config (edit config/concepts.json to change)
        self.scene_categories = _CONCEPTS["visual"]["scene_categories"]
        self.object_categories = _CONCEPTS["visual"]["object_categories"]
        
        # Encode all category texts
        self._encode_categories()
    
    def _encode_categories(self):
        """Pre-encode all category texts for efficiency"""
        texts = self.scene_categories + self.object_categories
        self.category_tokens = clip.tokenize(texts).to(self.device)
        
        with torch.no_grad():
            self.category_features = self.model.encode_text(self.category_tokens)
            self.category_features /= self.category_features.norm(dim=-1, keepdim=True)
    
    def extract_features(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Extract visual features from a single frame
        
        Args:
            frame: numpy array in BGR format (OpenCV default)
            
        Returns:
            Dictionary with features
        """
        # Safety checks
        if frame is None or frame.size == 0:
            logger.warning("Empty frame received, returning empty features")
            return {}
        
        # Ensure frame is uint8 (0-255)
        if frame.dtype != np.uint8:
            logger.debug(f"Converting frame dtype from {frame.dtype} to uint8")
            if frame.dtype.kind == 'f':
                # Assume float in [0,1] range, scale to [0,255]
                frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
            else:
                # Just cast (may clip values >255)
                frame = frame.astype(np.uint8)
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(frame_rgb)
        
        # Preprocess for CLIP
        image_input = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Get image features
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            # Classify scene and objects
            similarity = (image_features @ self.category_features.T).squeeze(0)
            
            # Get top predictions
            scene_scores = similarity[:len(self.scene_categories)]
            object_scores = similarity[len(self.scene_categories):]
            
        # Extract basic image properties
        brightness = np.mean(frame)
        contrast = np.std(frame)
        color_std = np.std(frame, axis=(0, 1))
        
        # Dominant colors (simplified)
        pixels = frame_rgb.reshape(-1, 3)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans.fit(pixels)
        dominant_colors = kmeans.cluster_centers_.astype(int).tolist()
        
        return {
            "clip_embedding": image_features.cpu().numpy().tolist()[0],
            "scene_predictions": {
                self.scene_categories[i]: float(scene_scores[i])
                for i in range(len(self.scene_categories))
            },
            "object_predictions": {
                self.object_categories[i]: float(object_scores[i])
                for i in range(len(self.object_categories))
            },
            "brightness": float(brightness),
            "contrast": float(contrast),
            "color_variance": color_std.tolist(),
            "dominant_colors": dominant_colors
        }
    
    def extract_shot_features(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """
        Aggregate features across a shot (multiple frames)
        
        Args:
            frames: List of frames from the shot
            
        Returns:
            Aggregated shot-level features
        """
        if not frames:
            return {}
        
        # Extract features for each frame
        frame_features = [self.extract_features(frame) for frame in frames]
        
        # Aggregate embeddings (average)
        avg_embedding = np.mean([f["clip_embedding"] for f in frame_features], axis=0)
        
        # Aggregate scene predictions (average)
        scene_keys = frame_features[0]["scene_predictions"].keys()
        avg_scene = {
            key: np.mean([f["scene_predictions"][key] for f in frame_features])
            for key in scene_keys
        }
        
        # Aggregate object predictions (average)
        obj_keys = frame_features[0]["object_predictions"].keys()
        avg_objects = {
            key: np.mean([f["object_predictions"][key] for f in frame_features])
            for key in obj_keys
        }
        
        # Temporal dynamics
        brightness_sequence = [f["brightness"] for f in frame_features]
        contrast_sequence = [f["contrast"] for f in frame_features]
        
        return {
            "clip_embedding": avg_embedding.tolist(),
            "scene_predictions": avg_scene,
            "object_predictions": avg_objects,
            "brightness_mean": float(np.mean(brightness_sequence)),
            "brightness_std": float(np.std(brightness_sequence)),
            "contrast_mean": float(np.mean(contrast_sequence)),
            "contrast_std": float(np.std(contrast_sequence)),
            "dominant_colors": frame_features[0]["dominant_colors"],  # Use first frame's colors
            "shot_duration": len(frames) / 25.0  # Assuming 25fps, adjust as needed
        }
    def extract_zero_shot_features(self, frame: np.ndarray, custom_concepts: List[str] = None) -> Dict[str, Any]:
        """
        Extract zero-shot features - detect ANY concept dynamically.
        If custom_concepts provided, detect those; otherwise auto-detect.
        """
        # Ensure proper format
        if frame.dtype != np.uint8:
            if frame.dtype.kind == 'f':
                frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        image_input = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        
        result = {
            "clip_embedding": image_features.cpu().numpy().tolist()[0],
        }
        
        # If custom concepts provided, evaluate them
        if custom_concepts:
            concept_scores = {}
            for concept in custom_concepts:
                text = f"a photo of {concept}"
                text_tokens = clip.tokenize([text]).to(self.device)
                text_features = self.model.encode_text(text_tokens)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                similarity = (image_features @ text_features.T).item()
                concept_scores[concept] = similarity
            
            result["zero_shot_concepts"] = concept_scores
        
        return result