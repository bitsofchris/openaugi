# main.py
import config
from sources import ObsidianSource
from atomic_extractor import SimpleLLMAtomicExtractor
from storage import AtomicIdeaStore


def main():
    # 1. Load Documents
    source = ObsidianSource(config.OBSIDIAN_VAULT_PATH)
    documents = source.load_documents()

    # 2. Extract atomic ideas from documents
    atomic_extractor = SimpleLLMAtomicExtractor()
    atomic_ideas = atomic_extractor.extract_atomic_ideas(documents)
    print(f"Extracted {len(atomic_ideas)} atomic ideas")

    store = AtomicIdeaStore(config.LANCEDB_PATH)
    store.save_atomic_ideas(atomic_ideas)

    # 3. Create embeddings for atomic ideas
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
