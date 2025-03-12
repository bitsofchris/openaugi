# main.py
import os

import config
from sources import ObsidianSource
from atomic_extractor import SimpleLLMAtomicExtractor
from storage import KnowledgeStore
from embedder import LlamaIndexEmbedder
from clusterer import UMAPHDBSCANClusterer
from visualizer import ClusterVisualizer


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

    # 6. Cluster atomic ideas
    print("Clustering atomic notes...")
    atomic_notes = store.get_all_atomic_notes()

    if atomic_notes:
        clusterer = UMAPHDBSCANClusterer(hdbscan_min_cluster_size=1)
        clusters = clusterer.cluster_documents(atomic_notes, store=store)
        print(f"Created {len(clusters)} clusters")

        # 7. Visualize clusters
        print("Creating cluster visualization...")
        visualizer = ClusterVisualizer(output_dir=os.path.join(config.OUTPUT_DIR, "visualizations"))
        viz_path = visualizer.visualize_clusters(clusters, title="Atomic Notes Clusters")
        print(f"Cluster visualization saved to {viz_path}")

        # Future: Implement distillation
        # distiller = LLMDistiller()
        # distilled_notes = distiller.distill_knowledge(clusters)
        # store.save_clean_notes(distilled_notes)

    print("Pipeline completed successfully")


if __name__ == "__main__":
    main()
