"""
Formatting utilities for the Voice Note Concept Extractor.
"""

import re
from typing import List, Dict, Any, Tuple


def highlight_text(
    text: str,
    start: int,
    end: int,
    highlight_color: str = "#FFFF00",
    opacity: float = 0.3,
) -> str:
    """
    Highlight a portion of text using HTML/CSS.

    Args:
        text: The full text
        start: Start position for highlighting
        end: End position for highlighting
        highlight_color: CSS color for highlighting
        opacity: Opacity for the highlight effect

    Returns:
        HTML-formatted text with highlighting
    """
    if not (0 <= start < len(text) and 0 < end <= len(text) and start < end):
        return text

    before = text[:start]
    highlight = text[start:end]
    after = text[end:]

    # Escape HTML entities in all parts
    before = before.replace("<", "&lt;").replace(">", "&gt;")
    highlight = highlight.replace("<", "&lt;").replace(">", "&gt;")
    after = after.replace("<", "&lt;").replace(">", "&gt;")

    # Convert newlines to <br> for proper HTML display
    before = before.replace("\n", "<br>")
    highlight = highlight.replace("\n", "<br>")
    after = after.replace("\n", "<br>")

    # Create the highlighted HTML
    result = f"{before}<span style='background-color: {highlight_color}; opacity: {opacity};'>{highlight}</span>{after}"
    return result


def get_section_boundaries(
    results: Dict[str, Any], section_index: int
) -> Tuple[str, Tuple[int, int]]:
    """
    Get the text and character boundaries for a specific section.

    Args:
        results: The extraction results
        section_index: Index of the section

    Returns:
        Tuple of (section_text, (start_position, end_position))
    """
    if section_index >= len(results["sections"]):
        return "", (0, 0)

    section = results["sections"][section_index]
    section_text = section["section_text"]

    # Calculate the start position in the full transcript
    full_transcript = ""
    start_pos = 0

    for i, s in enumerate(results["sections"]):
        if i < section_index:
            start_pos += len(s["section_text"])
        full_transcript += s["section_text"]

    end_pos = start_pos + len(section_text)

    return section_text, (start_pos, end_pos)


def format_timestamp(seconds: float) -> str:
    """
    Format seconds as MM:SS.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    return f"{minutes:02d}:{remaining_seconds:05.2f}"


def format_idea_as_card(idea: Dict[str, Any], include_details: bool = True) -> str:
    """
    Format an idea as an HTML card.

    Args:
        idea: The idea dictionary
        include_details: Whether to include additional details

    Returns:
        HTML string for the card
    """
    importance = idea.get("importance", "medium").lower()
    importance_colors = {"high": "#FF5252", "medium": "#FFBD69", "low": "#80CBC4"}
    color = importance_colors.get(importance, "#CCCCCC")

    html = f"""
    <div style="border-left: 4px solid {color}; padding: 10px; margin: 10px 0; border-radius: 4px; background-color: #FFFFFF; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <h4 style="margin: 0 0 5px 0;">{idea["title"]}</h4>
    """

    if include_details:
        html += f"""
        <p style="margin: 5px 0; color: #666;">{idea["description"]}</p>
        """

        if idea.get("related_concepts"):
            related = ", ".join(idea["related_concepts"])
            html += f"""
            <div style="margin-top: 5px; font-size: 0.9em; color: #888;">
                <strong>Related:</strong> {related}
            </div>
            """

    html += "</div>"
    return html


def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text to a maximum length with ellipsis.

    Args:
        text: The text to truncate
        max_length: Maximum length before truncation

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."
