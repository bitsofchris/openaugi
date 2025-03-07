from typing import List, Dict, Any, Union
from llama_index.core.schema import Document
from interfaces import DocumentSource, Embedder, Clusterer, Distiller
from models import DistilledNote


class DistillationPipeline:
    def __init__(
        self,
        source: DocumentSource,
        embedder: Embedder = None,
        clusterer: Clusterer = None,
        distiller: Distiller = None,
    ):
        self.source = source
        self.embedder = embedder
        self.clusterer = clusterer
        self.distiller = distiller

    def run(self) -> Union[List[Document], List[DistilledNote]]:
        # Load documents
        documents = self.source.load_documents()

        # If we have an embedder, embed the documents
        if self.embedder:
            documents = self.embedder.embed_documents(documents)
            print(f"Embedded {len(documents)} documents")

        # If we have a clusterer, cluster the documents
        clusters = []
        if self.clusterer and documents:
            clusters = self.clusterer.cluster_documents(documents)
            print(f"Created {len(clusters)} clusters")

        # If we have a distiller, distill the clusters
        distilled_notes = []
        if self.distiller and clusters:
            distilled_notes = self.distiller.distill_knowledge(clusters)
            print(f"Created {len(distilled_notes)} distilled notes")
            return distilled_notes

        # If we didn't distill, return the documents
        return documents


def create_pipeline_from_config(config: Dict[str, Any]) -> DistillationPipeline:
    # Create the source component
    source_config = config.get("source", {})
    source_path = source_config.get("path", "")
    source_specific_config = source_config.get("config", {})

    # For now, we only support ObsidianSource
    from sources import ObsidianSource

    source = ObsidianSource(source_path, source_specific_config)

    # For now, return a pipeline with just the source
    # We'll add other components as they're implemented
    return DistillationPipeline(source)
