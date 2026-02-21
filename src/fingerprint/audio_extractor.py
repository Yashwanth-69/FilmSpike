# At the very top of audio_extractor.py
import os
import sys
import logging
import tempfile
import numpy as np
from moviepy.editor import VideoFileClip
from faster_whisper import WhisperModel
import librosa

# Optional: import pkg_resources with graceful fallback
try:
    import pkg_resources
    PKG_RESOURCES_AVAILABLE = True
except ImportError:
    PKG_RESOURCES_AVAILABLE = False
    # If pkg_resources is missing, we can still proceed
    # but may need to handle subprocess issues differently
    logging.warning("pkg_resources not available - subprocess may have issues")

# Set PYTHONPATH for subprocesses (critical for faster-whisper)
os.environ['PYTHONPATH'] = os.pathsep.join(sys.path)


logger = logging.getLogger(__name__)

class AudioExtractor:
    """Extracts audio features using faster-whisper and librosa"""
    
    def __init__(self, model_size: str = "base", device: str = "cpu", compute_type: str = "int8"):
        """
        Initialize faster-whisper model.
        
        Args:
            model_size: Model size ("tiny", "base", "small", "medium", "large")
            device: "cpu" or "cuda"
            compute_type: "int8", "float16", etc.
        """
        logger.info(f"Loading faster-whisper model ({model_size}) on {device}...")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        
    def extract_audio(self, video_path: str, start_time: float = 0, end_time: float = None) -> dict:
        """
        Extract audio features from video segment.
        
        Args:
            video_path: Path to video file
            start_time: Start time in seconds
            end_time: End time in seconds (None = until end)
            
        Returns:
            Dictionary with transcription and audio features.
        """
        # Extract audio segment to temporary file
        with VideoFileClip(video_path) as video:
            # If end_time is None, use full duration
            if end_time is None:
                end_time = video.duration
            
            # Subclip the segment
            segment = video.subclip(start_time, end_time)
            
            # Skip if no audio
            if segment.audio is None:
                return {"has_audio": False}
            
            # Write to temporary WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                temp_path = tmp.name
                segment.audio.write_audiofile(temp_path, logger=None, verbose=False)
        
        try:
            # Transcribe with faster-whisper
            segments, info = self.model.transcribe(temp_path, beam_size=5)
            
            # Collect all segment texts and details
            transcript = " ".join([s.text for s in segments])
            segments_list = [
                {
                    "start": s.start,
                    "end": s.end,
                    "text": s.text
                }
                for s in segments
            ]
            
            # Extract audio features with librosa
            y, sr = librosa.load(temp_path, sr=None)
            
            # Tempo
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            
            # RMS energy (loudness)
            rms = librosa.feature.rms(y=y)[0]
            
            # Zero crossing rate (indicator of sound type)
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            
            # Clean up temp file
            os.unlink(temp_path)
            
            return {
                "has_audio": True,
                "transcript": transcript,
                "language": info.language,
                "language_probability": info.language_probability,
                "segments": segments_list,
                "audio_features": {
                    "tempo": float(tempo) if tempo is not None else 0.0,
                    "mean_spectral_centroid": float(np.mean(spectral_centroids)),
                    "mean_spectral_rolloff": float(np.mean(spectral_rolloff)),
                    "mean_rms": float(np.mean(rms)),
                    "mean_zero_crossing_rate": float(np.mean(zcr)),
                    "silence_ratio": float(np.mean(rms < 0.01))  # approximate silence
                }
            }
            
        except Exception as e:
            logger.error(f"Audio extraction failed: {e}")
            # Clean up temp file if it exists
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            return {"has_audio": False, "error": str(e)}