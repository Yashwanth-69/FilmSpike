import json
import torch
import clip
import numpy as np
from pathlib import Path
from PIL import Image
import logging
from typing import List, Dict, Any, Optional
import cv2

# Load concept config once at module level
_CONFIG_PATH = Path(__file__).parent / "config" / "concepts.json"
with _CONFIG_PATH.open("r", encoding="utf-8") as _f:
    _CONCEPTS = json.load(_f)
_ZS_CFG = _CONCEPTS["zero_shot"]

logger = logging.getLogger(__name__)

class ZeroShotDetector:
    """
    Dynamic zero-shot concept detection using CLIP.
    No fixed categories - detects whatever concepts you ask for.
    """
    
    def __init__(self, clip_model, clip_preprocess, device: str = "cpu"):
        """
        Initialize with existing CLIP model from visual_extractor.
        This avoids loading CLIP twice.
        """
        self.model = clip_model
        self.preprocess = clip_preprocess
        self.device = device
        
        # Load all config values from external JSON
        self.confidence_threshold = _ZS_CFG.get("confidence_threshold", 0.25)
        self.top_k_primary = _ZS_CFG.get("top_k_primary_concepts", 5)
        self.concept_groups_cfg = _ZS_CFG["concept_groups"]
        self.expansion_rules = _ZS_CFG.get("expansion_rules", {})
        
        # Prompt templates for CLIP zero-shot (research shows these matter significantly)
        self.templates = [
            "a photo of a {}",
            "a photo of {}",
            "a scene showing {}",
            "a movie frame with {}",
            "a shot containing {}",
            "{}"
        ]

        # We'll build concept lists dynamically
        self.concept_cache = {}
        
    def generate_relevant_concepts(self, shot_frames: List[np.ndarray]) -> List[str]:
        """
        Automatically generate relevant concepts based on what's in the shot.
        This is the key to moving beyond fixed categories.
        """
        # Use CLIP to get image features for this shot
        if not shot_frames:
            return []
        
        # Sample a few frames
        sample_frames = shot_frames[::max(1, len(shot_frames)//3)][:3]
        frame_features = []
        
        for frame in sample_frames:
            # Convert to PIL
            if frame.dtype != np.uint8:
                if frame.dtype.kind == 'f':
                    frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Preprocess and get features
            image_input = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = self.model.encode_image(image_input)
                features /= features.norm(dim=-1, keepdim=True)
                frame_features.append(features.cpu().numpy())
        
        if not frame_features:
            return []
        
        avg_features = np.mean(frame_features, axis=0)
        
        # Instead of fixed concepts, we'll use CLIP's zero-shot to identify what's present
        # This uses a broad set of "seed concepts" that expand dynamically
        seed_concepts = _ZS_CFG["seed_concepts"]
        
        # But we don't just use these directly - we use them as seeds
        # to find the most relevant concepts for THIS shot
        
        # Encode all seed concepts
        concept_texts = [f"a photo of a {c}" for c in seed_concepts]
        text_tokens = clip.tokenize(concept_texts).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.cpu().numpy()
        
        # Calculate similarities
        similarities = np.dot(avg_features, text_features.T)[0]
        
        # Get top concepts (those that actually appear)
        top_indices = np.argsort(similarities)[-10:][::-1]
        detected_concepts = [seed_concepts[i] for i in top_indices if similarities[i] > self.confidence_threshold]
        
        # Apply expansion rules from config
        expanded_concepts = list(detected_concepts)
        for concept in detected_concepts:
            for rule_key, expansions in self.expansion_rules.items():
                if rule_key in concept:
                    expanded_concepts.extend(expansions)
        
        # Remove duplicates and return
        return list(set(expanded_concepts))
    
    def detect_concepts(self, frame: np.ndarray, concepts: List[str]) -> Dict[str, float]:
        """
        Detect which concepts are present in a single frame.
        """
        if not concepts or frame is None or frame.size == 0:
            return {}
        
        # Ensure proper format
        if frame.dtype != np.uint8:
            if frame.dtype.kind == 'f':
                frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
        
        # Convert to PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Get image features
        image_input = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        
        # Use multiple templates for better detection [citation:1]
        concept_scores = {}
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            for concept in concepts:
                # Try multiple prompt templates
                template_scores = []
                for template in self.templates[:3]:  # Use a subset for speed
                    text = template.format(concept)
                    text_tokens = clip.tokenize([text]).to(self.device)
                    text_features = self.model.encode_text(text_tokens)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    
                    similarity = (image_features @ text_features.T).item()
                    template_scores.append(similarity)
                
                # Use max similarity across templates
                concept_scores[concept] = max(template_scores)
        
        return concept_scores
    
    def detect_concepts_in_shot(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """
        Detect concepts across an entire shot.
        """
        if not frames:
            return {"detected_concepts": {}, "primary_concepts": []}
        
        # First, dynamically determine what concepts to look for
        relevant_concepts = self.generate_relevant_concepts(frames)
        
        if not relevant_concepts:
            return {"detected_concepts": {}, "primary_concepts": []}
        
        # Sample frames for detection
        sample_frames = frames[::max(1, len(frames)//5)][:5]
        
        all_scores = []
        for frame in sample_frames:
            scores = self.detect_concepts(frame, relevant_concepts)
            all_scores.append(scores)
        
        # Aggregate across frames
        aggregated = {}
        concept_presence = {}
        
        for concept in relevant_concepts:
            concept_scores = [s.get(concept, 0) for s in all_scores]
            avg_score = np.mean(concept_scores)
            max_score = np.max(concept_scores)
            
            if max_score > self.confidence_threshold:  # Threshold loaded from config
                aggregated[concept] = {
                    "avg_confidence": float(avg_score),
                    "max_confidence": float(max_score),
                    "frames_detected": sum(1 for s in concept_scores if s > self.confidence_threshold)
                }
                concept_presence[concept] = float(avg_score)
        
        # Get primary concepts (top 5 by avg confidence)
        primary_concepts = sorted(
            concept_presence.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:self.top_k_primary]
        
        # Group concepts using config-driven groups
        concept_groups = {}
        grouped = set()
        for group_name, keywords in self.concept_groups_cfg.items():
            matched = [c for c in concept_presence if any(k in c for k in keywords)]
            if matched:
                concept_groups[group_name] = matched
                grouped.update(matched)
        
        # Anything not in a named group goes to "other"
        other = [c for c in concept_presence if c not in grouped]
        if other:
            concept_groups["other"] = other
        
        return {
            "detected_concepts": aggregated,
            "primary_concepts": [{"concept": c, "confidence": s} for c, s in primary_concepts],
            "concept_groups": {k: v for k, v in concept_groups.items() if v},
            "total_concepts_detected": len(concept_presence)
        }