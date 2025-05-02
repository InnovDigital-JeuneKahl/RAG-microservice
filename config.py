import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Vector database settings
VECTOR_DB_PATH = os.path.join(BASE_DIR, "vectordb")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Local embedding model
OLLAMA_BASE_URL = "http://localhost:11434"  # Default Ollama URL

# Default models
DEFAULT_LLM = "qwen2.5:3b-instruct"  # Default model for generation
ALTERNATIVE_LLM = "gemma3:4b"  # Alternative model

# RAG settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_KEYWORDS_PER_CHUNK = 15

# API settings
API_TITLE = "Transcription RAG System"
API_DESCRIPTION = "A RAG system for file transcriptions with metadata"
API_VERSION = "0.1.0"