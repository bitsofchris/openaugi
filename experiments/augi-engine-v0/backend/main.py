# main.py
import os

import config
from sources import ObsidianSource
from atomic_extractor import SimpleLLMAtomicExtractor
from storage import KnowledgeStore
from embedder import LlamaIndexEmbedder
from clusterer import UMAPHDBSCANClusterer
from visualizer import ClusterVisualizer, KnowledgeMapVisualizer
from distiller import ConceptDistiller
from selector import IntraClusterSimilarityFilter


def _process_in_chunks(documents, chunk_size=20):
    for i in range(0, len(documents), chunk_size):
        yield documents[i:i + chunk_size]


def main():
    # Initialize components
    source = ObsidianSource(config.OBSIDIAN_VAULT_PATH)
    store = KnowledgeStore(config.LANCE_DB_PATH)
    extractor = SimpleLLMAtomicExtractor()
    embedder = LlamaIndexEmbedder()
    distiller = ConceptDistiller()

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
        chunk_size = 20
        for chunk in _process_in_chunks(unprocessed_docs, chunk_size):
            # Extract atomic notes
            atomic_notes = extractor.extract_atomic_ideas(chunk)

            # Embed atomic notes
            embedded_notes = embedder.embed_documents(atomic_notes)

            # Save atomic notes
            store.save_atomic_notes(embedded_notes)
            print(f"Extracted and saved {len(embedded_notes)} atomic notes")

            # Mark documents as processed
            for doc in chunk:
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

        # 8. Knowledge distillation options
        print("Starting knowledge distillation...")

        # Choose distillation approach
        distillation_approach = config.DISTILLATION_METHOD

        # Get already distilled note IDs to avoid re-processing
        already_distilled_ids = store.get_already_distilled_note_ids()
        print(f"Found {len(already_distilled_ids)} atomic notes that have already been distilled")

        if distillation_approach == "similarity":
            # Option 1: Filter clusters for highly similar document groups
            print("Finding highly similar document groups within clusters...")
            similarity_filter = IntraClusterSimilarityFilter(similarity_threshold=0.85)
            similarity_groups = similarity_filter.filter_clusters(clusters)
            print(f"Found {len(similarity_groups)} similarity groups across all clusters")

            # Filter out groups where all notes have already been distilled
            groups_to_distill = []
            for group in similarity_groups:
                # Get IDs of notes in this group
                group_ids = [doc.doc_id for doc in group]

                # Check if any notes are new (not already distilled)
                new_notes = [doc_id for doc_id in group_ids if doc_id not in already_distilled_ids]

                if len(new_notes) >= 2:  # Only keep if at least 2 new notes
                    groups_to_distill.append(group)

            print(f"Filtered to {len(groups_to_distill)} similarity groups with new notes")
            distilled_notes = distiller.distill_knowledge(groups_to_distill)

        else:  # cluster approach
            # Option 2: Distill entire clusters
            print("Distilling knowledge from entire clusters...")

            # Filter clusters to only include those with new notes
            clusters_to_distill = []
            for cluster in clusters:
                # Get IDs of notes in this cluster
                cluster_ids = [doc.doc_id for doc in cluster]

                # Check if any notes are new (not already distilled)
                new_notes = [doc_id for doc_id in cluster_ids if doc_id not in already_distilled_ids]

                if len(new_notes) >= 2:  # Only keep if at least 2 new notes
                    clusters_to_distill.append(cluster)

            print(f"Filtered to {len(clusters_to_distill)} clusters with new notes")
            distilled_notes = distiller.distill_knowledge(clusters_to_distill)

        if distilled_notes:
            # 9. Embed distilled notes
            print("Embedding distilled notes...")
            distilled_docs = [note.doc for note in distilled_notes]
            embedded_distilled_docs = embedder.embed_documents(distilled_docs)

            # Map back to DistilledNote objects
            for i, doc in enumerate(embedded_distilled_docs):
                distilled_notes[i].doc = doc

            # 10. Save distilled notes
            print("Saving distilled notes...")
            # Create a mapping of note IDs to source atomic note IDs
            source_id_map = {note.doc_id: note.source_ids for note in distilled_notes}

            # Save using the KnowledgeStore
            distilled_docs = [note.doc for note in distilled_notes]
            store.save_clean_notes(distilled_docs, source_id_map)
            print(f"Saved {len(distilled_notes)} distilled notes")

        # Create updated knowledge map with distilled notes
        print("Creating updated knowledge map with distilled notes...")
        visualizer = KnowledgeMapVisualizer(output_dir=os.path.join(config.OUTPUT_DIR, "visualizations"))
        viz_path = visualizer.visualize_knowledge_base(store, include_connections=True)
        print(f"Updated knowledge map saved to {viz_path}")

    print("Pipeline completed successfully")


if __name__ == "__main__":
    main()
