"""
Configuration settings for Augi application.
"""

OBSIDIAN_VAULT_PATH = "/Users/chris/zk-copy-for-testing"  # Path to Obsidian vault

# LLM Model Settings
DEFAULT_LLM_MODEL = "gpt-4o-mini-2024-07-18"  # Default model for extraction

# File Processing
SUPPORTED_EXTENSIONS = [".md", ".txt"]
MAX_CHUNK_SIZE = 8000  # Maximum characters per chunk for processing

# Embedding Settings
EMBEDDING_MODEL = "text-embedding-3-small"  # Default embedding model
EMBEDDING_DIMENSION = 1536  # Dimension of embeddings

# Vector Database Settings
LANCE_DB_PATH = "data/vector_db"  # Path to store vector database files
SIMILARITY_THRESHOLD = 0.75  # Threshold for considering notes similar

# UI Settings
MAX_DISPLAY_CHARS = 300  # Maximum characters to display in UI previews
CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence to highlight an idea
