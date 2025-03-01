"""
Configuration settings for the Voice Note Concept Extractor application.
"""

import os
from pathlib import Path

DEBUG_MODE = True

# File paths
APP_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = APP_DIR / "data"
TEMP_DIR = APP_DIR / "temp"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# File names
DEFAULT_RESULTS_FILE = "extracted_ideas.json"
DEFAULT_TRANSCRIPT_FILE = "transcript.txt"

# UI settings
PAGE_TITLE = "Voice Note Concept Extractor"
PAGE_ICON = "🧠"
PRIMARY_COLOR = "#4A6FA5"
SECONDARY_COLOR = "#FF5252"
ACCENT_COLOR = "#4CAF50"
BACKGROUND_COLOR = "#F5F8FA"

# Graph settings
NODE_COLORS = {
    "high": "#FF5252",  # Red
    "medium": "#FFBD69",  # Orange
    "low": "#80CBC4",  # Teal
}
GRAPH_BG_COLOR = "#222222"
GRAPH_FONT_COLOR = "white"
GRAPH_HEIGHT = "600px"
GRAPH_WIDTH = "100%"

# LLM settings
DEFAULT_LLM_MODEL = "gpt-4o-mini-2024-07-18"
DEFAULT_TEMPERATURE = 0.1


# Session state keys
class SessionKeys:
    """Keys for accessing session state variables."""

    RESULTS = "results"
    TRANSCRIPT = "transcript"
    FILE_PROCESSED = "file_processed"
    PLAYBACK_ACTIVE = "playback_active"
    PLAYBACK_POSITION = "playback_position"
    REVEAL_SPEED = "reveal_speed"
    REVEALED_IDEAS = "revealed_ideas"
    SELECTED_IDEA = "selected_idea"
    HIGHLIGHTED_SECTION = "highlighted_section"
    CURRENT_PHASE = "current_phase"
