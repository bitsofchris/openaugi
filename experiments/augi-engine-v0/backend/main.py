# main.py
import config
from sources import ObsidianSource
from atomic_extractor import SimpleLLMAtomicExtractor
from storage import KnowledgeStore
from embedder import LlamaIndexEmbedder


def main():
    # Initialize components
    source = ObsidianSource(config.OBSIDIAN_VAULT_PATH)
    store = KnowledgeStore(config.LANCE_DB_PATH)
    extractor = SimpleLLMAtomicExtractor()
    embedder = LlamaIndexEmbedder()

    # 1. Load raw documents and set IDs properly
    raw_documents = source.load_documents()
    print(f"Loaded {len(raw_documents)} source documents")

    # 2. Filter to only new documents
    new_documents = store.get_new_documents(raw_documents)
    print(f"Found {len(new_documents)} new documents")

    # 3. Process new documents
    if new_documents:
        # Embed and save new documents
        embedded_docs = embedder.embed_documents(new_documents)
        store.save_raw_documents(embedded_docs)
        print(f"Processed and saved {len(embedded_docs)} new documents")

    # 4. Get documents needing atomic extraction
    unprocessed_docs = store.get_unprocessed_documents()
    print(f"Found {len(unprocessed_docs)} documents needing atomic extraction")

    # 5. Extract and save atomic notes
    if unprocessed_docs:
        # Extract atomic notes
        atomic_notes = extractor.extract_atomic_ideas(unprocessed_docs)

        # Embed atomic notes
        embedded_notes = embedder.embed_documents(atomic_notes)

        # Save atomic notes
        store.save_atomic_notes(embedded_notes)
        print(f"Extracted and saved {len(embedded_notes)} atomic notes")

        # Mark documents as processed
        for doc in unprocessed_docs:
            store.mark_raw_document_processed(doc.doc_id)

    print("Pipeline completed successfully")

    # At this point, we have:
    # - Raw documents saved with embeddings
    # - Atomic notes extracted, embedded, and saved
    # - Relationships maintained between raw and atomic notes

    # Future steps (not implemented here):
    # 9. Cluster atomic notes to find related concepts
    # 10. Generate clean/distilled notes from clusters
    # 11. Embed and save clean notes
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


if __name__ == "__main__":
    main()
