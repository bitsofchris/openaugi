"""
File I/O utilities for the Voice Note Concept Extractor.
"""

import os
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, Union, BinaryIO


def save_uploaded_file(
    uploaded_file: BinaryIO, directory: Union[str, Path] = None
) -> str:
    """
    Save an uploaded file to a temporary location.

    Args:
        uploaded_file: The uploaded file object
        directory: Optional directory to save to (uses system temp if None)

    Returns:
        Path to the saved file
    """
    if directory:
        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        return file_path
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name


def load_transcript(file_path: Union[str, Path]) -> str:
    """
    Load a transcript from a file.

    Args:
        file_path: Path to the transcript file

    Returns:
        The transcript text
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def save_results(results: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """
    Save extraction results to a JSON file.

    Args:
        results: The results dictionary
        file_path: Path to save the JSON file
    """
    directory = os.path.dirname(file_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def load_results(file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    Load extraction results from a JSON file.

    Args:
        file_path: Path to the JSON file

    Returns:
        The results dictionary or None if file doesn't exist
    """
    if not os.path.exists(file_path):
        return None

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


# hardcoded for testing
# TODO
def load_extracted_ideas(file_path="app/data/extracted_ideas.json"):
    """
    Loads the extracted ideas JSON file and returns the data as a dictionary.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find file: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data
