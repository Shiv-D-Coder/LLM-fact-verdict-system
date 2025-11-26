# config.py
# Configuration and constants

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")

# Model Configuration
EMBEDDING_MODEL = "text-embedding-3-small"  #Use "text-embedding-3-large(3072 dim) for large KB
LLM_MODEL = "gpt-4o"  
EMBEDDING_DIMENSIONS = 1536

# Retrieval Configuration
TOP_K_RETRIEVAL = 5
SIMILARITY_THRESHOLD = 0.3  # Minimum cosine similarity to consider

# File Paths
DATA_DIR = "data"
KB_DIR = os.path.join(DATA_DIR, "KB")
FACTS_CSV = os.path.join(DATA_DIR, "trusted_facts.csv")
INDEX_DIR = "vector_store"
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "faiss_index.bin")
METADATA_PATH = os.path.join(INDEX_DIR, "metadata.pkl")

# SpaCy Configuration
SPACY_MODEL = "en_core_web_sm"  # Lightweight model, can upgrade to "en_core_web_lg"

# Logging
LOG_LEVEL = "INFO"
