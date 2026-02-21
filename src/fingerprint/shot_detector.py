import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple
import cv2

logger = logging.getLogger(__name__)

class ShotDetector:
    """Detects shot boundaries using PySceneDetect (primary) or OpenCV fallback"""
    
    def __init__(self, use_fallback: bool = False):
        self.use_fallback = use_fallback
        if not use_fallback:
            try:
                from scenedetect import SceneManager, open_video
                from scenedetect.detectors import ContentDetector
                self.scene_manager = SceneManager()
                self.scene_manager.add_detector(ContentDetector(threshold=30.0))
                self.open_video = open_video
                logger.info("PySceneDetect loaded successfully")
            except ImportError:
                logger.warning("PySceneDetect not available, using OpenCV fallback")
                self.use_fallback = True
    
    def detect_shots(self, video_path: Path, threshold: float = 30.0) -> List[Tuple[float, float]]:
        """
        Detect shot boundaries in video
        
        Args:
            video_path: Path to video file
            threshold: Sensitivity threshold (lower = more sensitive)
            
        Returns:
            List of (start_time, end_time) tuples for each shot
        """
        if not self.use_fallback:
            return self._detect_with_scenedetect(video_path, threshold)
        else:
            return self._detect_with_opencv(video_path, threshold)
    
    def _detect_with_scenedetect(self, video_path: Path, threshold: float) -> List[Tuple[float, float]]:
        """Use PySceneDetect for accurate shot detection"""
        from scenedetect import SceneManager, open_video
        from scenedetect.detectors import ContentDetector
        
        video = self.open_video(str(video_path))
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=threshold))
        scene_manager.detect_scenes(video)
        scenes = scene_manager.get_scene_list()
        
        shot_boundaries = [(scene[0].get_seconds(), scene[1].get_seconds()) for scene in scenes]
        logger.info(f"Detected {len(shot_boundaries)} shots with PySceneDetect")
        return shot_boundaries
    
    def _detect_with_opencv(self, video_path: Path, threshold: float) -> List[Tuple[float, float]]:
        """Fallback method using histogram differences"""
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        prev_hist = None
        shot_boundaries = []
        shot_start = 0
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to grayscale and compute histogram
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            cv2.normalize(hist, hist)
            
            if prev_hist is not None:
                # Chi-square distance
                diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CHISQR)
                if diff > threshold:
                    # Shot boundary
                    shot_end = frame_count / fps
                    shot_boundaries.append((shot_start, shot_end))
                    shot_start = shot_end
            
            prev_hist = hist
            frame_count += 1
        
        # Add final shot
        shot_end = frame_count / fps
        shot_boundaries.append((shot_start, shot_end))
        
        cap.release()
        logger.info(f"Detected {len(shot_boundaries)} shots with OpenCV fallback")
        return shot_boundaries