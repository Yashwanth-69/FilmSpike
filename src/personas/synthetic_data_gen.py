import csv
import logging
import random
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = _ROOT / "data" / "personas"
OUTPUT_FILE = OUTPUT_DIR / "synthetic_reddit_comments.csv"

TARGET_MOVIES = ["Transformers", "Pacific Rim", "Godzilla 2000"]

# We engineer ground-truth templates that simulate vocabulary and tone differences
TEMPLATES = {
    1: { # Analytical Sci-Fi
        "name": "Analytical Sci-Fi",
        "phrases": [
            "The world-building in {movie} is genuinely fascinating if you look past the surface.",
            "I appreciated how {movie} handled the scale and physics of the mecha designs.",
            "Say what you will about the plot, but the lore and the neural drift concepts in {movie} are top-tier sci-fi.",
            "The CGI holds up because the director understood weight, mass, and practical lighting.",
            "It's a masterclass in visual storytelling, particularly how the scale of the threat is established early on.",
            "{movie} does a surprisingly good job of integrating the sci-fi tech into a believable near-future economy.",
            "If you analyze the mechanical engineering represented in {movie}, it's surprisingly grounded."
        ]
    },
    2: { # Casual Blockbuster
        "name": "Casual Blockbuster",
        "phrases": [
            "bro {movie} is literal fire, just turn your brain off and enjoy the explosions 🔥",
            "I saw {movie} in IMAX and the bass literally shook my seat. Insane movie.",
            "Giant robots fighting giant monsters. What more do you want from {movie}??",
            "the action sequences in {movie} are absolutely goated.",
            "10/10 popcorn flick. {movie} never gets old.",
            "I don't care about the plot holes, {movie} is pure hype.",
            "Watching {movie} makes me want to punch a wall, the soundtrack goes so hard."
        ]
    },
    3: { # Meme/Troll
        "name": "Meme/Troll",
        "phrases": [
            "me watching {movie}: haha robot go brrrrr 😂",
            "is {movie} a jojo reference???",
            "tbh {movie} was mid, they should have just used a nuke.",
            "when the giant robot did the thing in {movie} I literally ascended 💀",
            "this says a lot about society.",
            "POV: you're trying to watch {movie} but the plot is missing.",
            "I rate {movie} 5/7 perfect score."
        ]
    },
    4: { # Character Drama
        "name": "Character Drama",
        "phrases": [
            "I honestly teared up during {movie}. The relationship between the pilots is the emotional core.",
            "People talk about the action, but {movie} is really a story about trauma and finding human connection.",
            "The actor's chemistry in {movie} was incredible. You could really feel their shared pain.",
            "The romance subplot in {movie} was surprisingly tender amidst all the chaos.",
            "I wish they spent more time on the characters in {movie}, their backstory was heartbreaking.",
            "The true focus of {movie} is how tragedy unites humanity, not just the fighting.",
            "That final sacrifice scene in {movie} broke me emotionally."
        ]
    },
    0: { # Noise
        "name": "Noise",
        "phrases": [
            "Anyone else here playing the new Zelda game instead?",
            "honestly pizza is overrated. don't @ me.",
            "This is just a test to see if my Reddit bot is working. Beep boop.",
            "Did you guys hear about the stock market crash?",
            "First!",
            "I lost my car keys today and I am so mad.",
            "Check out my soundcloud link in bio!"
        ]
    }
}

def generate_dataset(target_count_per_cluster: int = 50):
    """Generate the full synthetic dataset across all movies and clusters."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    post_id_counter = 1000
    
    # 15% noise
    noise_count = int(target_count_per_cluster * 4 * 0.15)
    
    for movie in TARGET_MOVIES:
        for cluster_id in range(1, 5):
            for _ in range(target_count_per_cluster):
                # Pick a random template and add a little random noise to it
                base_text = random.choice(TEMPLATES[cluster_id]["phrases"]).format(movie=movie)
                
                # Suffix variations to prevent identical hashes
                suffix = random.choice(["", " Totally agree.", " Imho.", " Just saying.", " 🤷‍♂️", " :)"])
                text = base_text + suffix
                
                records.append({
                    "post_id": f"p_{post_id_counter}",
                    "subreddit": "r/movies",
                    "comment_text": text,
                    "upvotes": random.randint(-5, 500) if cluster_id != 3 else random.randint(-50, 50),
                    "timestamp": int(time.time() - random.randint(0, 86400 * 30)),
                    "comment_length": len(text),
                    "movie_reference": movie,
                    "true_cluster_id": cluster_id
                })
                post_id_counter += 1
                
        # Generate Noise (0)
        for _ in range(max(1, noise_count // len(TARGET_MOVIES))):
            base_text = random.choice(TEMPLATES[0]["phrases"])
            text = base_text + random.choice(["", " lol.", " anyone?", " sigh."])
            
            records.append({
                "post_id": f"p_{post_id_counter}",
                "subreddit": random.choice(["r/all", "r/funny", "r/AskReddit"]),
                "comment_text": text,
                "upvotes": random.randint(-100, 10),
                "timestamp": int(time.time() - random.randint(0, 86400 * 30)),
                "comment_length": len(text),
                "movie_reference": "None",
                "true_cluster_id": 0
            })
            post_id_counter += 1
            
    random.shuffle(records)
    
    fieldnames = ["post_id", "subreddit", "comment_text", "upvotes", "timestamp", "comment_length", "movie_reference", "true_cluster_id"]
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
        
    logger.info(f"Successfully generated {len(records)} structurally realistic synthetic comments to {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_dataset(target_count_per_cluster=50)
