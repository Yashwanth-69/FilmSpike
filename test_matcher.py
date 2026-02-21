import json
from src.analysis.keyword_aggregator import KeywordAggregator
from src.analysis.movie_matcher import MovieMatcher

with open("data/output/test.json", "r") as f:
    fingerprint = json.load(f)

aggregator = KeywordAggregator()
kw_result = aggregator.aggregate(fingerprint)
film_vector = kw_result.get("film_vector")

matcher = MovieMatcher()
similar = matcher.find_similar(
    fingerprint,
    film_vector=film_vector,
    top_k=10,
    genre_hints=kw_result["film_analysis"].get("genre_hints", []),
    top_keywords=kw_result.get("top_keywords", [])
)

for m in similar:
    print(f"{m['title']} ({m['year']}) - Score: {m['similarity_score']:.4f} [Vis: {m['_visual_score']:.4f}, Kw: {m['_keyword_score']:.4f}]")
