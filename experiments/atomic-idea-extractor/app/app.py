import time
import streamlit as st
from utils.file_io import load_extracted_ideas, load_transcript
from services.markdown_exporter import note_to_markdown


def main():
    st.title("Voice Distillation Prototype")

    # Initialize session state if not already set.
    if "mode" not in st.session_state:
        st.session_state.mode = (
            "transcript"  # default mode is to show the raw transcript
        )

    # Sidebar with two buttons: Record and Distill.
    st.sidebar.header("Actions")
    if st.sidebar.button("Record"):
        st.session_state.mode = "transcript"
    if st.sidebar.button("Distill"):
        st.session_state.mode = "distill"

    # Load the raw transcript.
    raw_text = load_transcript(
        "/Users/chris/repos/augi-private/app/data/transcript.txt"
    )

    # If we're in distill mode, load the atomic notes.
    if st.session_state.mode == "distill":
        try:
            data = load_extracted_ideas()
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return
        atomic_notes = data.get("deduplicated_ideas", [])
    else:
        atomic_notes = []

    # Create two columns: left for the raw transcript / spinner and right for the atomic notes.
    col1, col2 = st.columns(2)

    # Left Column: Either show raw transcript or the spinner/progress.
    with col1:
        if st.session_state.mode == "transcript":
            st.header("Raw Transcript")
            st.markdown(raw_text)
        elif st.session_state.mode == "distill":
            st.header("Distillation in Progress")
            left_container = st.empty()
            spinner_url = "https://i.gifer.com/ZZ5H.gif"  # example spinner GIF URL
            # Start with a starting message.
            left_container.markdown(
                f"""
                <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 200px;">
                    <img src="{spinner_url}" alt="Loading spinner" style="width: 100px;"/>
                    <p>Starting distillation...</p>
                </div>
            """,
                unsafe_allow_html=True,
            )

    # Right Column: Show a placeholder in transcript mode, or the distilled atomic notes in distill mode.
    with col2:
        if st.session_state.mode == "transcript":
            st.header("Atomic Notes")
            st.markdown("Atomic notes will appear here once you click 'Distill'.")
        elif st.session_state.mode == "distill":
            st.header("Distilled Atomic Notes")
            total_notes = len(atomic_notes)
            # Iterate over each note, displaying one at a time.
            for idx, note in enumerate(atomic_notes):
                time.sleep(1)  # Simulate processing delay.
                with st.expander(note.get("title", f"Untitled {idx}")):
                    # Combine description, importance, and related concepts into one editable text area.
                    description = note.get("description", "")
                    importance = note.get("importance", "N/A")
                    related = note.get("related_concepts", [])
                    default_text = f"{description}\n\n" f"#{importance}\n\n" + " ".join(
                        f"[[{rc}]]" for rc in related
                    )
                    edited_text = st.text_area(
                        "Edit note:", value=default_text, key=f"text_{idx}", height=200
                    )
                    md_content = note_to_markdown(note, edited_text)
                    st.download_button(
                        label="Export to Obsidian",
                        data=md_content,
                        file_name=f"{note.get('title', f'Note_{idx}')}.md",
                        mime="text/markdown",
                        key=f"download_{idx}",
                    )
                # Update the left column progress after each note.
                left_container.markdown(
                    f"""
                    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 200px;">
                        <img src="{spinner_url}" alt="Loading spinner" style="width: 100px;"/>
                        <p>Distilling... {idx + 1} / {total_notes} notes extracted.</p>
                    </div>
                """,
                    unsafe_allow_html=True,
                )
            # Final update once distillation is complete.
            left_container.markdown(
                f"""
                <div style="text-align: center; margin-top: 50px;">
                    <h3>Distillation complete! {total_notes} notes extracted.</h3>
                </div>
            """,
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
