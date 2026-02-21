#!/usr/bin/env python3
"""
Command-line interface for Phase 1 fingerprinting
"""

import argparse
import json
from pathlib import Path
import sys
import logging

from .pipeline import FilmFingerprinter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Film Content Fingerprinting System")
    subparsers = parser.add_subparsers(dest="command", help="Commands", required=True)
    
    # Fingerprint command
    fingerprint_parser = subparsers.add_parser("fingerprint", help="Generate fingerprint for a film")
    fingerprint_parser.add_argument("video_path", help="Path to video file")
    fingerprint_parser.add_argument("--output", "-o", help="Output JSON file path")
    fingerprint_parser.add_argument("--fps", type=float, default=2.0, help="Target FPS for frame extraction (default: 2)")
    fingerprint_parser.add_argument("--title", help="Film title (for metadata)")
    fingerprint_parser.add_argument("--year", type=int, help="Release year")
    fingerprint_parser.add_argument("--genre", help="Genre (comma-separated)")
    
    args = parser.parse_args()
    
    if args.command == "fingerprint":
        video_path = Path(args.video_path)
        if not video_path.exists():
            logger.error(f"Video not found: {video_path}")
            sys.exit(1)
        
        # Determine output path if not specified
        if args.output:
            output_path = args.output
        else:
            output_path = video_path.stem + "_fingerprint.json"
        
        # Generate fingerprint
        fingerprinter = FilmFingerprinter()
        fingerprint = fingerprinter.fingerprint(str(video_path), output_path, target_fps=args.fps)
        
        # Print summary
        print("\n=== Fingerprint Summary ===")
        print(f"Film: {video_path.name}")
        print(f"Duration: {fingerprint['video_info']['duration']:.2f}s")
        print(f"Shots detected: {fingerprint['video_info']['shot_count']}")
        print(f"Dominant scenes: {fingerprint['summary']['dominant_scenes']}")
        print(f"Dominant emotions: {fingerprint['summary']['dominant_emotions']}")
        print(f"Face presence ratio: {fingerprint['summary']['face_presence_ratio']:.2f}")
        print(f"Audio presence ratio: {fingerprint['summary']['audio_presence_ratio']:.2f}")
        print(f"\nFingerprint saved to: {output_path}")
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()