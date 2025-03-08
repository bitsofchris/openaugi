# main.py
import config
from sources import ObsidianSource
from atomic_extractor import SimpleLLMAtomicExtractor
from storage import AtomicIdeaStore


def main():
    source = ObsidianSource(config.OBSIDIAN_VAULT_PATH)
    store = AtomicIdeaStore(config.LANCE_DB_PATH)
    extractor = SimpleLLMAtomicExtractor()

    # 1. Load Documents
    documents = source.load_documents()
    print(f"Loaded {len(documents)} documents")

    # 2. Extract atomic ideas (will skip already processed documents)
    new_atomic_ideas = extractor.extract_atomic_ideas(documents, store)
    print(f"Extracted new {len(new_atomic_ideas)} atomic ideas")

    # 3. Save new atomic ideas to LanceDB (if any)
    if new_atomic_ideas:
        store.save_atomic_ideas(new_atomic_ideas)

    # 4. Get all atomic ideas for further processing
    all_atomic_ideas = store.get_all_atomic_ideas_as_documents()
    print(f"Total atomic ideas available: {len(all_atomic_ideas)}")

    # 5. Add embeddings


    # TODO: Implement Embedder
    # embedder = OpenAIEmbedder()
    # embedded_ideas = embedder.embed_documents(atomic_ideas)

    # 4. Save embeddings and metadata to LanceDB
    # TODO: Implement storage

    # 5. Cluster atomic ideas by embedding
    # TODO: Implement Clusterer
    # clusterer = KMeansClusterer(n_clusters=10)
    # clusters = clusterer.cluster_documents(embedded_ideas)
    # print(f"Created {len(clusters)} clusters")

    # 6. Distill/Summarize clusters into higher level ideas
    # TODO: Implement Distiller
    # distiller = LLMDistiller()
    # distilled_notes = distiller.distill_knowledge(clusters)
    # print(f"Created {len(distilled_notes)} distilled notes")

    # 7. Visualize the map of ideas
    # TODO: Implement visualization

    print("Pipeline completed successfully")


if __name__ == "__main__":
    main()
