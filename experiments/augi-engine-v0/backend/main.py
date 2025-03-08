# main.py
import config
from sources import ObsidianSource
from atomic_extractor import SimpleLLMAtomicExtractor
from storage import AtomicIdeaStore
from embedder import LlamaIndexEmbedder


def main():
    source = ObsidianSource(config.OBSIDIAN_VAULT_PATH)
    store = AtomicIdeaStore(config.LANCE_DB_PATH)
    extractor = SimpleLLMAtomicExtractor()
    embedder = LlamaIndexEmbedder()

    # 1. Load Documents
    documents = source.load_documents()
    print(f"Loaded {len(documents)} source documents")
    for doc in documents:
        doc.metadata["is_atomic_idea"] = False

    # 2. Extract atomic ideas (will skip already processed documents)
    new_atomic_ideas = extractor.extract_atomic_ideas(documents, store)
    print(f"Extracted new {len(new_atomic_ideas)} atomic ideas")

    # 3. Save new atomic ideas to LanceDB (if any)
    if new_atomic_ideas:
        store.save_atomic_ideas(new_atomic_ideas)

    # 4. Get all documents needing embeddings and embed them, save
    docs_needing_embeddings = store.get_documents_without_embeddings()
    for doc in documents:
        if not store.has_processed_document(doc.doc_id):
            docs_needing_embeddings.append(doc)
    print(f"Found {len(docs_needing_embeddings)} documents needing embeddings")
    if docs_needing_embeddings:
        embedded_docs = embedder.embed_documents(docs_needing_embeddings)
        store.save_documents_with_embeddings(embedded_docs)



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
