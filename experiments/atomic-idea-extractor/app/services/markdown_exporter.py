def note_to_markdown(note, edited_text=None):
    """
    Convert a note dictionary to a Markdown formatted string.
    If edited_text is provided, it replaces the note's original description.
    """
    title = note.get("title", "Untitled")
    # Use the edited text if available; otherwise, use the original description.
    description = (
        edited_text if edited_text is not None else note.get("description", "")
    )
    importance = note.get("importance", "N/A")
    related = note.get("related_concepts", [])

    md_text = f"# {title}\n\n"
    md_text += f"**Description:**\n{description}\n\n"
    md_text += f"**Importance:** {importance}\n\n"
    if related:
        md_text += "**Related Concepts:** " + ", ".join(related) + "\n\n"
    return md_text
