import json
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
from typing import Dict, Any, List
import cv2

from .video_loader import VideoLoader
from .shot_detector import ShotDetector
from .visual_extractor import VisualExtractor
from .audio_extractor import AudioExtractor
from .face_extractor import FaceExtractor
from .emotion_extractor import EmotionExtractor
from .zero_shot_detector import ZeroShotDetector
from src.analysis.keyword_aggregator import KeywordAggregator
from src.analysis.movie_matcher import MovieMatcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FilmFingerprinter:
    """
    Main orchestrator for Phase 1: Multimodal Content Fingerprinting
    
    Combines all modules to create comprehensive film fingerprint
    """
    
    def __init__(self, device: str = None):
        self.device = device
        self.shot_detector = ShotDetector()
        self.visual_extractor = VisualExtractor(device=device)
        self.audio_extractor = AudioExtractor()
        self.face_extractor = FaceExtractor()
        self.emotion_extractor = EmotionExtractor(device=device)
        self.zero_shot_detector = ZeroShotDetector(
            self.visual_extractor.model,
            self.visual_extractor.preprocess,
            self.device
        )
        logger.info("Zero-shot detector initialized")
        logger.info("FilmFingerprinter initialized with all modules")
    
    def _update_progress(self, status: str, percent: float):
        progress_file = Path("data/output/progress.json")
        try:
            progress_file.parent.mkdir(parents=True, exist_ok=True)
            with open(progress_file, "w") as f:
                json.dump({"status": status, "percent": percent}, f)
        except Exception as e:
            logger.error(f"Failed to update progress: {e}")

    def fingerprint(self, video_path: str, output_path: str = None, target_fps: float = 2.0) -> Dict[str, Any]:
        """
        Generate complete fingerprint for a video.
        
        Args:
            video_path: Path to video file
            output_path: Optional path to save fingerprint JSON
            target_fps: Target frames per second for extraction (default 2)
            
        Returns:
            Dictionary containing complete film fingerprint
        """
        video_path = Path(video_path)
        logger.info(f"Starting fingerprinting for: {video_path.name}")
        self._update_progress("Initializing Video Loader...", 5)
        
        # Step 1: Load video and get basic info
        with VideoLoader(str(video_path), target_fps=target_fps) as loader:
            video_info = {
                "filename": video_path.name,
                "duration": loader.duration,
                "original_fps": loader.original_fps,
                "total_frames": loader.total_frames,
                "effective_fps": loader.effective_fps
            }
            
            # Step 2: Detect shots
            logger.info("Detecting shot boundaries...")
            self._update_progress("Detecting Shot Boundaries...", 10)
            shots = self.shot_detector.detect_shots(video_path)
            video_info["shot_count"] = len(shots)
            
            # Step 3: Extract all frames first (with sampling)
            logger.info("Extracting frames...")
            self._update_progress("Extracting Frames...", 20)
            frames = []
            timestamps = []
            for frame, ts in loader.extract_frames():
                frames.append(frame)
                timestamps.append(ts)
            
            frames = np.array(frames, dtype=object)  # Keep as list of arrays
            timestamps = np.array(timestamps)
            
            # Step 4: Process each shot
            logger.info("Processing shots...")
            self._update_progress("Starting Shot Processing...", 30)
            shots_data = []
            
            total_shots = len(shots)
            for shot_idx, (shot_start, shot_end) in enumerate(tqdm(shots, desc="Processing shots")):
                # Find frames in this shot
                shot_frames_idx = np.where((timestamps >= shot_start) & (timestamps <= shot_end))[0]
                shot_frames = [frames[i] for i in shot_frames_idx]
                
                if len(shot_frames) == 0:
                    # No frames in this shot (maybe shot too short? skip)
                    continue
                
                # Extract visual features
                visual_features = self.visual_extractor.extract_shot_features(shot_frames)
                
                # Extract audio features for this shot
                audio_features = self.audio_extractor.extract_audio(
                    str(video_path), 
                    shot_start, 
                    shot_end
                )
                
                # Extract face information
                face_info = self.face_extractor.detect_faces_in_shot(shot_frames)
                
                # Extract emotional arc
                emotion_info = self.emotion_extractor.analyze_shot_emotions(shot_frames)
                zero_shot_info = self.zero_shot_detector.detect_concepts_in_shot(shot_frames)
                
                # Combine shot data
                shot_data = {
                    "shot_index": shot_idx,
                    "start_time": shot_start,
                    "end_time": shot_end,
                    "duration": shot_end - shot_start,
                    "frame_count": len(shot_frames),
                    "visual_features": visual_features,
                    "audio_features": audio_features,
                    "face_info": face_info,
                    "emotion_info": emotion_info,
                    "zero_shot_concepts": zero_shot_info
                }
                
                shots_data.append(shot_data)
                
                if total_shots > 0:
                    current_percent = 30 + (50 * (shot_idx + 1) / total_shots) # Scaling from 30% to 80%
                    self._update_progress(f"Processing Shot {shot_idx+1}/{total_shots}", round(current_percent))
            
            # Step 5: Create film-level fingerprint
            fingerprint = {
                "video_info": video_info,
                "shots": shots_data,
                "metadata": {
                    "fingerprint_version": "1.0",
                    "generated_at": str(np.datetime64('now')),
                    "modules_used": ["shot_detector", "visual_extractor", "audio_extractor", 
                                     "face_extractor", "emotion_extractor"]
                }
            }
            
            # Add film-level aggregations (Phase 1 summary)
            fingerprint["summary"] = self._generate_summary(shots_data)

            # ── Phase 2: Keyword synthesis & movie matching ───────────────────
            logger.info("Running Phase 2: keyword aggregation...")
            self._update_progress("Running Keyword Aggregation...", 85)
            try:
                aggregator = KeywordAggregator()
                kw_result = aggregator.aggregate(fingerprint)
                fingerprint["top_keywords"] = kw_result["top_keywords"]
                fingerprint["film_analysis"] = kw_result["film_analysis"]
                film_vector = kw_result.get("film_vector")

                if film_vector:
                    logger.info("Running Phase 2: movie matching via ChromaDB...")
                    self._update_progress("Matching Similar Movies...", 90)
                    matcher = MovieMatcher()
                    similar = matcher.find_similar(
                        fingerprint,
                        film_vector=film_vector,
                        top_k=10,
                        genre_hints=kw_result["film_analysis"].get("genre_hints", []),
                        top_keywords=kw_result.get("top_keywords", []),
                    )
                    fingerprint["similar_movies"] = similar
                    logger.info(f"Found {len(similar)} similar movies")
                else:
                    fingerprint["similar_movies"] = []
                    logger.warning("No film vector produced — skipping movie matching")
            except Exception as e:
                logger.error(f"Phase 2 failed: {e}", exc_info=True)
                fingerprint["top_keywords"] = []
                fingerprint["similar_movies"] = []
                fingerprint["film_analysis"] = {}

            # Save if output path provided
            if output_path:
                with open(output_path, 'w') as f:
                    json.dump(fingerprint, f, indent=2, default=self._json_serialize)
                logger.info(f"Fingerprint saved to: {output_path}")

            self._update_progress("Complete", 100)
            return fingerprint
    
    def _generate_summary(self, shots_data: List[Dict]) -> Dict[str, Any]:
        """Generate film-level summary statistics"""
        if not shots_data:
            return {}
        
        # Aggregate across shots
        total_duration = sum(s["duration"] for s in shots_data)
        
        # Dominant scene types
        all_scenes = []
        for shot in shots_data:
            scenes = shot["visual_features"].get("scene_predictions", {})
            if scenes:
                top_scene = max(scenes, key=scenes.get)
                all_scenes.append(top_scene)
        
        from collections import Counter
        scene_distribution = Counter(all_scenes)
        
        # Dominant emotions
        all_emotions = []
        for shot in shots_data:
            emotion = shot["emotion_info"].get("dominant_emotion", "unknown")
            all_emotions.append(emotion)
        
        emotion_distribution = Counter(all_emotions)
        
        # Face presence
        shots_with_faces = sum(1 for s in shots_data if s["face_info"].get("has_faces", False))
        
        # Audio presence
        shots_with_audio = sum(1 for s in shots_data if s["audio_features"].get("has_audio", False))
        
        return {
            "total_duration": total_duration,
            "shots_count": len(shots_data),
            "avg_shot_duration": total_duration / len(shots_data) if shots_data else 0,
            "dominant_scenes": dict(scene_distribution.most_common(5)),
            "dominant_emotions": dict(emotion_distribution.most_common(5)),
            "face_presence_ratio": shots_with_faces / len(shots_data) if shots_data else 0,
            "audio_presence_ratio": shots_with_audio / len(shots_data) if shots_data else 0
        }
    
    def _json_serialize(self, obj):
        """Helper for JSON serialization of numpy types"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        raise TypeError(f"Type {type(obj)} not serializable")