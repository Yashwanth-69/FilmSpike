import cv2
import numpy as np
from typing import Generator, Tuple, Optional
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoLoader:
    """Handles video loading and frame extraction with intelligent sampling"""
    
    def __init__(self, video_path: str, target_fps: Optional[float] = None):
        """
        Initialize video loader
        
        Args:
            video_path: Path to video file
            target_fps: Desired frames per second for extraction (None = original)
        """
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        self.cap = cv2.VideoCapture(str(video_path))
        self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.original_fps if self.original_fps > 0 else 0
        
        # Calculate sampling rate
        if target_fps and target_fps < self.original_fps:
            self.sample_interval = int(self.original_fps / target_fps)
            self.effective_fps = target_fps
        else:
            self.sample_interval = 1
            self.effective_fps = self.original_fps
            
        logger.info(f"Video loaded: {self.video_path.name}")
        logger.info(f"Duration: {self.duration:.2f}s, FPS: {self.original_fps}, Frames: {self.total_frames}")
        logger.info(f"Sampling: 1 frame every {self.sample_interval} frames ({self.effective_fps} FPS)")
    
    def extract_frames(self, max_frames: Optional[int] = None) -> Generator[Tuple[np.ndarray, float], None, None]:
        """
        Extract frames with intelligent sampling
        
        Args:
            max_frames: Maximum number of frames to extract (None = all)
            
        Yields:
            Tuple of (frame as numpy array, timestamp in seconds)
        """
        frame_count = 0
        extracted_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Sample at calculated interval
            if frame_count % self.sample_interval == 0:
                timestamp = frame_count / self.original_fps
                yield frame, timestamp
                extracted_count += 1
                
                if max_frames and extracted_count >= max_frames:
                    break
                    
            frame_count += 1
            
        logger.info(f"Extracted {extracted_count} frames from {frame_count} total frames")
    
    def release(self):
        """Release video capture"""
        self.cap.release()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()