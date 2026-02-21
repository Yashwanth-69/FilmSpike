"""
Quick smoke test for Phase 2 using existing test.json
Run from project root: python src/tests/test_phase2.py
"""
import json
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

def main():
    fp_path = ROOT / "data" / "output" / "test.json"
    if not fp_path.exists():
        fp_path = ROOT / "sample_fingerprint.json"

    print(f"\nLoading fingerprint from: {fp_path}")
    with open(fp_path, "r") as f:
        fingerprint = json.load(f)

    print(f"  Shots: {len(fingerprint.get('shots', []))}")

    # Test KeywordAggregator
    from src.analysis.keyword_aggregator import KeywordAggregator
    agg = KeywordAggregator()
    result = agg.aggregate(fingerprint)

    print(f"\n✅ KeywordAggregator OK")
    print(f"  top_keywords: {result['top_keywords'][:5]}")
    print(f"  film_analysis: {result['film_analysis']}")
    assert len(result["top_keywords"]) > 0, "top_keywords is empty!"
    assert result["film_vector"] is not None, "film_vector is None!"
    assert len(result["film_vector"]) == 512, f"Expected 512-dim vector, got {len(result['film_vector'])}"

    # Test MovieMatcher
    from src.analysis.movie_matcher import MovieMatcher
    matcher = MovieMatcher()
    count_before = matcher.collection.count()
    print(f"\n✅ ChromaDB connected — {count_before} movies")

    similar = matcher.find_similar(
        fingerprint,
        film_vector=result["film_vector"],
        top_k=5,
        genre_hints=result["film_analysis"].get("genre_hints", []),
    )

    print(f"\n✅ MovieMatcher OK — found {len(similar)} similar movies")
    for m in similar:
        print(f"  {m['title']} ({m['year']}) {m['genres']} — score: {m['similarity_score']}")

    assert len(similar) > 0, "No similar movies returned!"
    assert "title" in similar[0] and "similarity_score" in similar[0]

    print("\n🎉 All Phase 2 tests passed!\n")


if __name__ == "__main__":
    main()
