import config

from llama_index.readers.obsidian import ObsidianReader


def _parse_documents(file_path: str, reader_object) -> str:
    documents = reader_object.load_data()
    print(f"Found {len(documents)} documents in {file_path}")


def main():
    # Load Documents
    folder_path = config.OBSIDIAN_VAULT_PATH
    obsidian_reader = ObsidianReader(
        folder_path, extract_tasks=True, remove_tasks_from_text=True
    )
    _parse_documents(folder_path, obsidian_reader)

    # Skip for now - embeddings and saving them for full documents

    # Extract atomic ideas (link to source doc)

    # Create Embeddings of atomic ideas and Save embeddings, note, and metadata to lancedb

    # Cluster atomic ideas by embedding

    # Visualize clusters

    # Distill/Summarize clusters into higher level ideas

    # Visualize the new map of ideas


if __name__ == "__main__":
    main()
