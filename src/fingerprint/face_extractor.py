import cv2
import numpy as np
import logging
from typing import List, Dict, Any
import insightface

logger = logging.getLogger(__name__)

class FaceExtractor:
    """Detects and analyzes faces using InsightFace"""
    
    def __init__(self):
        logger.info("Loading InsightFace model...")
        # Use CPU; if you have GPU, set ctx_id appropriately
        self.model = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        self.model.prepare(ctx_id=0)  # ctx_id=0 for GPU, -1 for CPU
        logger.info("InsightFace model loaded.")
        
    def detect_faces(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in a single frame.
        
        Args:
            frame: numpy array in BGR format (OpenCV default)
            
        Returns:
            List of face dictionaries with bounding boxes, landmarks, age, gender, embedding.
        """
        # Ensure frame is uint8 (0-255)
        if frame is None or frame.size == 0:
            return []
        
        if frame.dtype != np.uint8:
            logger.debug(f"Converting frame dtype from {frame.dtype} to uint8 for face detection")
            if frame.dtype.kind == 'f':
                # Assume float in [0,1] range, scale to [0,255]
                frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
        
        faces = self.model.get(frame)
        
        results = []
        for face in faces:
            # Bounding box: [x1, y1, x2, y2]
            bbox = face.bbox.astype(int).tolist()
            
            # Landmarks (5 points: eyes, nose, mouth corners)
            landmarks = face.landmark_2d_106 if hasattr(face, 'landmark_2d_106') else None
            if landmarks is not None:
                landmarks = landmarks.tolist()
            
            # Age, gender, embedding
            age = int(face.age) if hasattr(face, 'age') else None
            gender = 'Male' if face.gender == 1 else 'Female' if hasattr(face, 'gender') else None
            embedding = face.embedding.tolist() if hasattr(face, 'embedding') else None
            
            results.append({
                "bbox": bbox,
                "det_score": float(face.det_score),
                "landmarks": landmarks,
                "age": age,
                "gender": gender,
                "embedding": embedding
            })
        
        return results
        
    def detect_faces_in_shot(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """
        Detect and track faces across a shot (multiple frames).
        
        Args:
            frames: List of frames from the shot.
            
        Returns:
            Aggregated face information for the shot.
        """
        all_faces = []
        face_counts = []
        
        for frame in frames:
            faces = self.detect_faces(frame)
            all_faces.extend(faces)
            face_counts.append(len(faces))
        
        if not all_faces:
            return {
                "has_faces": False,
                "face_count_mean": 0.0,
                "face_count_max": 0,
                "total_faces_detected": 0
            }
        
        # Aggregate statistics
        ages = [f["age"] for f in all_faces if f["age"] is not None]
        genders = [f["gender"] for f in all_faces if f["gender"] is not None]
        
        return {
            "has_faces": True,
            "face_count_mean": float(np.mean(face_counts)),
            "face_count_max": int(max(face_counts)),
            "total_faces_detected": len(all_faces),
            "avg_detection_score": float(np.mean([f["det_score"] for f in all_faces])),
            "age_estimates": ages,
            "gender_distribution": {
                "male": sum(1 for g in genders if g == "Male"),
                "female": sum(1 for g in genders if g == "Female")
            } if genders else {},
            # Optionally include average embedding? Not needed for now.
        }