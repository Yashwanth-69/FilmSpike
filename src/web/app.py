import os
import json
import logging
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import sys
_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(_ROOT))

from src.fingerprint.pipeline import FilmFingerprinter
from src.analysis.movie_matcher import MovieMatcher
from src.web.persona_service import WebPersonaService

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static")
CORS(app)

WEB_DIR = _ROOT / "src" / "web"
CONCEPTS_FILE = _ROOT / "src" / "fingerprint" / "config" / "concepts.json"
OUTPUT_DIR = _ROOT / "data" / "output"
LATEST_FINGERPRINT = OUTPUT_DIR / "test.json"
UPLOADS_DIR = _ROOT / "data" / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

persona_service = WebPersonaService()

@app.route("/")
def index():
    return send_from_directory(WEB_DIR / "templates", "index.html")

@app.route("/api/upload", methods=["POST"])
def upload_and_fingerprint():
    """Takes configuring data and a video file, updates concepts.json, and runs fingerprint pipeline."""
    try:
        if 'video' not in request.files:
            return jsonify({"error": "No video file provided"}), 400
            
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({"error": "No selected file"}), 400
            
        # Reset progress tracker immediately
        progress_file = OUTPUT_DIR / "progress.json"
        progress_file.parent.mkdir(parents=True, exist_ok=True)
        with open(progress_file, "w") as f:
            json.dump({"status": "Uploading...", "percent": 0}, f)
            
        # Optional Concepts Overrides
        concepts_data_str = request.form.get('concepts')
        if concepts_data_str:
            new_concepts = json.loads(concepts_data_str)
            
            # Read existing concepts to preserve the "zero_shot" config structure
            if CONCEPTS_FILE.exists():
                with open(CONCEPTS_FILE, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
            else:
                existing_data = {"visual": {}, "emotion": {}}
                
            # Update the specific nested lists depending on what the UI sent
            if "visual" not in existing_data: existing_data["visual"] = {}
            if "emotion" not in existing_data: existing_data["emotion"] = {"categories": ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]}
            if "object_categories" in new_concepts:
                existing_data["visual"]["object_categories"] = new_concepts["object_categories"]
            if "scene_categories" in new_concepts:
                existing_data["visual"]["scene_categories"] = new_concepts["scene_categories"]
            
            # The top-level ones remain top-level
            if "custom_actions" in new_concepts:
                existing_data["custom_actions"] = new_concepts["custom_actions"]
            if "custom_themes" in new_concepts:
                existing_data["custom_themes"] = new_concepts["custom_themes"]
            
            with open(CONCEPTS_FILE, "w", encoding="utf-8") as f:
                json.dump(existing_data, f, indent=4)
            logger.info("Merged concepts.json with user input.")

        # Save uploaded video
        video_path = str(UPLOADS_DIR / video_file.filename)
        video_file.save(video_path)
        
        logger.info(f"Starting fingerprint process on {video_path}...")
        
        # Initialize fingerprinter and run
        fingerprinter = FilmFingerprinter()
        fingerprint_data = fingerprinter.fingerprint(video_path, output_path=str(LATEST_FINGERPRINT))
        
        # The fingerprint pipeline saves test.json natively, but we can also return success
        return jsonify({
            "message": "Fingerprint successfully generated.",
            "duration": fingerprint_data["video_info"].get("duration", 0),
            "top_keywords": fingerprint_data.get("top_keywords", [])
        }), 200

    except Exception as e:
        logger.error(f"Error during upload/fingerprint: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route("/api/progress", methods=["GET"])
def get_progress():
    """Returns the current fingerprint processing percentage."""
    progress_file = OUTPUT_DIR / "progress.json"
    if not progress_file.exists():
        return jsonify({"status": "Waiting to start...", "percent": 0}), 200
    try:
        with open(progress_file, "r") as f:
            data = json.load(f)
        return jsonify(data), 200
    except Exception:
        return jsonify({"status": "Reading status...", "percent": 0}), 200

@app.route("/api/check_status", methods=["GET"])
def check_status():
    """Tells the frontend if a fingerprint is already completed and ready for analysis."""
    if LATEST_FINGERPRINT.exists():
        return jsonify({"has_fingerprint": True}), 200
    return jsonify({"has_fingerprint": False}), 200

@app.route("/api/similar", methods=["GET"])
def get_similar_movies():
    """Reads latest fingerprint and finds top 10 similar movies."""
    try:
        if not LATEST_FINGERPRINT.exists():
            return jsonify({"error": "No fingerprint found. Please analyze a video first."}), 404
            
        with open(LATEST_FINGERPRINT, "r", encoding="utf-8") as f:
            fingerprint = json.load(f)
            
        film_vector = fingerprint.get("film_vector")
        if not film_vector:
            # Fallback to re-aggregating if missing
            from src.analysis.keyword_aggregator import KeywordAggregator
            agg = KeywordAggregator()
            kw_res = agg.aggregate(fingerprint)
            film_vector = kw_res.get("film_vector")
            
        matcher = MovieMatcher()
        similar = matcher.find_similar(
            fingerprint,
            film_vector=film_vector,
            top_k=10,
            genre_hints=fingerprint.get("film_analysis", {}).get("genre_hints", []),
            top_keywords=fingerprint.get("top_keywords", [])
        )
        
        return jsonify({"similar_movies": similar}), 200
        
    except Exception as e:
        logger.error(f"Error finding similar movies: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route("/api/personas", methods=["GET"])
def generate_personas():
    """Uses nuanced dataset to find target audience affinity and demographics."""
    try:
        if not LATEST_FINGERPRINT.exists():
            return jsonify({"error": "No fingerprint found."}), 404
            
        with open(LATEST_FINGERPRINT, "r", encoding="utf-8") as f:
            fingerprint = json.load(f)
            
        results = persona_service.calculate_affinity(fingerprint)
        return jsonify({"personas": results}), 200
        
    except Exception as e:
        logger.error(f"Error generating personas: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    logger.info("Starting Film Intelligence Web Server...")
    # Serve static files properly
    app.run(host="127.0.0.1", port=5000, debug=True)
