import config
from pipeline import create_pipeline_from_config


def main():
    # Create a simple configuration
    pipeline_config = {
        "source": {
            "type": "obsidian",
            "path": config.OBSIDIAN_VAULT_PATH,
            "config": {"extract_tasks": True, "remove_tasks_from_text": True},
        }
    }

    # Create and run the pipeline
    pipeline = create_pipeline_from_config(pipeline_config)
    documents = pipeline.run()

    # Print some info about the results
    print(f"Pipeline completed with {len(documents)} resulting documents")

    # Later steps will be added incrementally:
    # Skip for now - embeddings and saving them for full documents
    # - Extract atomic ideas (link to source doc)
    # - Create Embeddings of atomic ideas and Save embeddings & metadata to lancedb
    # - Cluster atomic ideas by embedding
    # - Visualize clusters
    # - Distill/Summarize clusters into higher level ideas
    # - Visualize the new map of ideas


if __name__ == "__main__":
    main()
