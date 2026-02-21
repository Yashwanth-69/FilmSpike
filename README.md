# Film Intelligence Pipeline 🎬🧠

A full-stack, AI-powered web application for deep analysis of movie trailers, audience demographics, and film similarity matching.

## Overview
This application takes an uploaded movie trailer (`.mp4`) and automatically:
1. **Extracts Visuals & Audio**: Uses OpenAI's `CLIP` to understand the scene composition and zero-shot objects present.
2. **Analyzes Emotion**: Scans actor faces for emotional tone mapping throughout the trailer.
3. **Similarity Engine**: Uses `ChromaDB` offline to mathematically find the Top 10 similar Hollywood blockbusters based on visual and conversational aesthetics.
4. **Persona Generation**: Embeds synthetic audience data via `SentenceTransformers` and pings a local `Ollama` (llama3) LLM daemon to dynamically recommend the exact Subreddits to target for marketing.

---

## 🛠️ GitHub Deployment & Packaging Guide

If you clone this project from GitHub onto a new computer, follow these exact steps to restore its full functionality.

### 1. Prerequisites 
- **Python 3.10+**
- **Ollama**: You must install [Ollama](https://ollama.com/) locally. 
  - After installing Ollama, run: `ollama pull llama3` in your terminal to download the language model.

### 2. Quick Setup
Open your terminal in the cloned repository folder and run the following commands:

**Windows System:**
```powershell
# 1. Create a virtual environment
python -m venv venv

# 2. Activate it
venv\Scripts\activate

# 3. Install all exact dependencies
pip install -r requirements.txt
```

**Mac/Linux System:**
```bash
# 1. Create a virtual environment
python3 -m venv venv

# 2. Activate it
source venv/bin/activate

# 3. Install all exact dependencies
pip install -r requirements.txt
```

### 3. Launching the Server
Once the virtual environment is built and dependencies are installed, just turn on the Flask backend:
```bash
python src/web/app.py
```

Then, open your browser and navigate to: **`http://127.0.0.1:5000`**

### What is NOT pushed to GitHub?
Because GitHub has file-size limits, the `.gitignore` safely excludes:
1. Your massive Virtual Environment (`venv/`)
2. The heavy `ChromaDB` embedded database (`data/chroma_db/`)
3. The uploaded `.mp4` video files
4. The downloaded HuggingFace models

**Don't worry!** Simply installing the `requirements.txt` fixes the code environment. The very first time you click "Analyze Similarity", the Python script will automatically rebuild your `.chromadb` from the `movies_fallback.csv` file natively! And the machine learning transformers (`clip`, `sentence-transformer`) will automatically redownload their weights the first time you run a video!
