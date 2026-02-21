import csv
import logging
import random
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = _ROOT / "data" / "personas"
OUTPUT_FILE = OUTPUT_DIR / "nuanced_reddit_comments.csv"

# Broad audience movies for our synthetic comments
TARGET_MOVIES = ["Transformers", "Pacific Rim", "Godzilla 2000", "Dune", "Interstellar"]

# Highly nuanced, specific categories
TEMPLATES = {
    1: { 
        "name": "VFX & Animation Enthusiasts",
        "phrases": [
            "The CGI rendering on the metal surfaces in {movie} is industry-leading.",
            "I kept pausing {movie} just to look at the particle effects during the explosions.",
            "Say what you want, but the VFX team for {movie} deserves an Oscar. The compositing is flawless.",
            "The way they handled the physics simulations of the destruction in {movie} is breathtaking.",
            "I work in Blender, and the animation weight in {movie} makes me want to quit. It's too good.",
            "The fluid dynamics in the water scenes of {movie} look insanely expensive.",
            "You can really tell they didn't cut corners on the ILM render farm for {movie}."
        ]
    },
    2: { 
        "name": "Cinematography & Directing Nerds",
        "phrases": [
            "The low-angle tracking shots in {movie} establish scale so beautifully.",
            "The director of photography for {movie} understood the assignment with the neon lighting contrast.",
            "That unbroken continuous panning shot in the middle of {movie} is a cinematic marvel.",
            "I love how the color-grading in {movie} shifts from cool blues to warm oranges as the act progresses.",
            "The blocking and framing in {movie} makes the chaotic scenes surprisingly legible.",
            "It's rare to see a blockbuster like {movie} use practical lighting on set so effectively.",
            "The cinematic composition in {movie} elevates it beyond a standard popcorn flick."
        ]
    },
    3: { 
        "name": "Mecha & Lore Fans",
        "phrases": [
            "The world-building and history they established for the factions in {movie} is so deep.",
            "I love that the mech designs in {movie} actually make logical engineering sense.",
            "Reading the lore wiki for {movie} makes the background details in the movie 10x better.",
            "The neural-drift concepts and machine integration logic in {movie} is my favorite trope.",
            "The sheer scale of the world and the history of the conflicts in {movie} is staggering.",
            "If you look at the schematics they show in {movie} for 2 seconds, they are completely accurate.",
            "I need a 50-hour RPG set in the universe of {movie} right now."
        ]
    },
    4: { 
        "name": "Military & Tactical Action Fans",
        "phrases": [
            "The combined arms tactics deployed by the human forces in {movie} is surprisingly realistic.",
            "I appreciated the accurate use of air-support and artillery formations in {movie}.",
            "Usually movies get military comms wrong, but the radio chatter in {movie} was spot on.",
            "The sheer firepower and strategic withdrawal scenes in {movie} were heavily grounded.",
            "Seeing modern infantry try to adapt to a threat like that in {movie} was strategically fascinating.",
            "They actually used proper target painting and laser designation in {movie} before the airstrike.",
            "The logistics of moving those weapons around in {movie} was handled very realistically."
        ]
    },
    5: { 
        "name": "Sound Design Junkies",
        "phrases": [
            "The sub-bass frequency when the core charges in {movie} literally rattled my teeth.",
            "The foley work for the grinding metal and servos in {movie} is audio perfection.",
            "If you don't watch {movie} on a solid surround sound system, you are missing half the movie.",
            "The sheer dynamic range in the audio mix of {movie} is incredible.",
            "The mechanical thud of the footsteps in {movie} is the best sound effect of the decade.",
            "The way they mixed the score into the chaotic soundscape in {movie} was masterful.",
            "That metallic screeching sound design in {movie} gave me goosebumps."
        ]
    },
    6: { 
        "name": "General Popcorn Audience",
        "phrases": [
            "bro {movie} is literal fire, just turn your brain off and enjoy the explosions 🔥",
            "10/10 popcorn flick. {movie} never gets old.",
            "I don't care about the plot holes, {movie} is pure hype.",
            "Watching {movie} makes me want to punch a wall, the soundtrack goes so hard.",
            "the action sequences in {movie} are absolutely goated.",
            "Giant robots fighting giant monsters. What more do you want from {movie}??",
            "I saw {movie} in IMAX and the bass literally shook my seat. Insane movie."
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

# Add Demographics to Map the Personas to for insights
DEMOGRAPHICS = {
    1: {"age_range": "18-35", "top_region": "North America", "gender_split": "60M/40F", "income": "High"},
    2: {"age_range": "25-45", "top_region": "Europe", "gender_split": "50M/50F", "income": "Medium"},
    3: {"age_range": "15-30", "top_region": "Asia", "gender_split": "80M/20F", "income": "Low"},
    4: {"age_range": "30-55", "top_region": "North America", "gender_split": "90M/10F", "income": "Medium"},
    5: {"age_range": "20-40", "top_region": "Global", "gender_split": "70M/30F", "income": "Medium"},
    6: {"age_range": "13-40", "top_region": "Global", "gender_split": "60M/40F", "income": "Varied"},
    0: {"age_range": "Unknown", "top_region": "Unknown", "gender_split": "Unknown", "income": "Unknown"}
}

def generate_dataset(target_count_per_cluster: int = 40):
    """Generate nuanced synthetic dataset."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    post_id_counter = 1000
    
    # 10% noise
    noise_count = int(target_count_per_cluster * 6 * 0.10)
    
    for movie in TARGET_MOVIES:
        for cluster_id in range(1, 7):
            for _ in range(target_count_per_cluster):
                base_text = random.choice(TEMPLATES[cluster_id]["phrases"]).format(movie=movie)
                suffix = random.choice(["", " Totally agree.", " Imho.", " Just saying.", " 🤷‍♂️", " :)", " 💯"])
                text = base_text + suffix
                
                records.append({
                    "post_id": f"p_{post_id_counter}",
                    "subreddit": "r/movies",
                    "comment_text": text,
                    "upvotes": random.randint(10, 2000),
                    "timestamp": int(time.time() - random.randint(0, 86400 * 30)),
                    "comment_length": len(text),
                    "movie_reference": movie,
                    "true_cluster_id": cluster_id,
                    "cluster_name": TEMPLATES[cluster_id]["name"],
                    "age_demographic": DEMOGRAPHICS[cluster_id]["age_range"],
                    "gender_demographic": DEMOGRAPHICS[cluster_id]["gender_split"],
                    "region": DEMOGRAPHICS[cluster_id]["top_region"]
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
                "true_cluster_id": 0,
                "cluster_name": "Noise",
                "age_demographic": "Unknown",
                "gender_demographic": "Unknown",
                "region": "Unknown"
            })
            post_id_counter += 1
            
    random.shuffle(records)
    
    fieldnames = ["post_id", "subreddit", "comment_text", "upvotes", "timestamp", "comment_length", "movie_reference", "true_cluster_id", "cluster_name", "age_demographic", "gender_demographic", "region"]
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
        
    logger.info(f"Successfully generated {len(records)} nuanced synthetic comments to {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_dataset(target_count_per_cluster=40)
