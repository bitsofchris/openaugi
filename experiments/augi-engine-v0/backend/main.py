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

    # 1. Load raw documents from source
    raw_documents = source.load_documents()
    print(f"Loaded {len(raw_documents)} source documents")

    # Mark as raw documents
    for doc in raw_documents:
        doc.metadata["is_raw_document"] = True
        doc.metadata["file_path"] = doc.metadata.get("file_path", "")

    # 2. Embed raw documents
    embedded_raw_docs = embedder.embed_documents(raw_documents)
    print(f"Embedded {len(embedded_raw_docs)} raw documents")

    # 3. Save embedded raw documents to database
    raw_doc_ids = store.save_raw_documents(embedded_raw_docs)
    print(f"Saved {len(raw_doc_ids)} raw documents to database")

    # 4. Identify documents that haven't been processed into atomic notes
    unprocessed_docs = []
    for i, doc_id in enumerate(raw_doc_ids):
        if not store.has_processed_document(doc_id):
            unprocessed_docs.append(embedded_raw_docs[i])

    print(f"Found {len(unprocessed_docs)} documents needing processing")

    # 5. Extract atomic notes from unprocessed documents
    if unprocessed_docs:
        atomic_notes = extractor.extract_atomic_ideas(unprocessed_docs)
        print(f"Extracted {len(atomic_notes)} atomic notes")

        # 6. Embed atomic notes
        embedded_atomic_notes = embedder.embed_documents(atomic_notes)
        print(f"Embedded {len(embedded_atomic_notes)} atomic notes")

        # 7. Save embedded atomic notes to database
        atomic_note_ids = store.save_atomic_notes(embedded_atomic_notes)
        print(f"Saved {len(atomic_note_ids)} atomic notes to database")

        # 8. Mark raw documents as processed
        for doc in unprocessed_docs:
            store.mark_raw_document_processed(doc.doc_id)

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

    print("Pipeline completed successfully")


if __name__ == "__main__":
    main()
