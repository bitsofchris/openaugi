"""
Extractor service for the Voice Note Concept Extractor.
Wraps the core VoiceNoteExtractor functionality.
"""

import os
from typing import Dict, Any, Union
from pathlib import Path

from voice_note_extractor import VoiceNoteExtractor
from utils.file_io import load_transcript, save_results, load_results
import config


class ExtractorService:
    """Service for handling voice note extraction."""

    def __init__(self, model: str = config.DEFAULT_LLM_MODEL):
        """
        Initialize the extractor service.

        Args:
            model: The LLM model to use
        """
        self.extractor = VoiceNoteExtractor(llm_model=model)

    def process_file(
        self, file_path: Union[str, Path], save_path: Union[str, Path] = None
    ) -> Dict[str, Any]:
        """
        Process a transcript file and extract ideas.

        Args:
            file_path: Path to the transcript file
            save_path: Optional path to save results

        Returns:
            Dictionary of extraction results
        """
        # Load transcript
        transcript_text = load_transcript(file_path)

        # Process with extractor
        results = self.extractor.process_transcript(transcript_text)

        # Save results if path provided
        if save_path:
            save_results(results, save_path)

        return results

    def map_ideas_to_sections(self, results: Dict[str, Any]) -> Dict[int, list]:
        """
        Create a mapping from section indices to their ideas.

        Args:
            results: Extraction results

        Returns:
            Dictionary mapping section indices to lists of idea indices
        """
        section_to_ideas = {}
        idea_index = 0

        for section_idx, section in enumerate(results["sections"]):
            section_ideas = []
            for _ in section["ideas"]:
                section_ideas.append(idea_index)
                idea_index += 1

            section_to_ideas[section_idx] = section_ideas

        return section_to_ideas

    def get_idea_by_index(self, results: Dict[str, Any], index: int) -> Dict[str, Any]:
        """
        Get an idea by its global index in all_ideas.

        Args:
            results: Extraction results
            index: Global index of the idea

        Returns:
            The idea dictionary or None if index is invalid
        """
        if 0 <= index < len(results["all_ideas"]):
            return results["all_ideas"][index]
        return None

    def get_section_for_idea(self, results: Dict[str, Any], idea_index: int) -> int:
        """
        Find which section contains a given idea.

        Args:
            results: Extraction results
            idea_index: Global index of the idea

        Returns:
            Section index or -1 if not found
        """
        idea_count = 0
        for section_idx, section in enumerate(results["sections"]):
            section_idea_count = len(section["ideas"])
            if idea_index < idea_count + section_idea_count:
                return section_idx
            idea_count += section_idea_count
        return -1
