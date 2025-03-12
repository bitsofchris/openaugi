# interfaces/pipeline.py
from abc import ABC, abstractmethod
from typing import List
from llama_index.core.schema import Document


class DocumentSource(ABC):
    @abstractmethod
    def load_documents(self) -> List[Document]:
        pass


class AtomicExtractor(ABC):
    @abstractmethod
    def extract_atomic_ideas(self, documents: List[Document]) -> List[Document]:
        """Extract atomic ideas from documents, linking back to source docs"""
        pass


class Embedder(ABC):
    @abstractmethod
    def embed_documents(self, documents: List[Document]) -> List[Document]:
        pass


class Clusterer(ABC):
    @abstractmethod
    def cluster_documents(self, documents: List[Document]) -> List[List[Document]]:
        pass


class DistilledNote:
    def __init__(self, doc: Document, source_ids: List[str]):
        """
        Initialize a distilled note with source document IDs.

        Args:
            doc: The Document object containing the distilled content
            source_ids: List of source document IDs that contributed to this note
        """
        self.doc = doc
        self.source_ids = source_ids

    @property
    def doc_id(self) -> str:
        return self.doc.doc_id


class Distiller(ABC):
    @abstractmethod
    def distill_knowledge(self, clusters: List[List[Document]]) -> List[DistilledNote]:
        """
        Distill/summarize clusters into higher-level ideas.

        Args:
            clusters: List of document clusters (each cluster is a list of Documents)

        Returns:
            List of DistilledNote objects
        """
        pass


# # Orchestrator
# class DistillationPipeline:
#     def __init__(
#         self,
#         source: DocumentSource,
#         embedder: Embedder,
#         clusterer: Clusterer,
#         distiller: Distiller,
#     ):
#         self.source = source
#         self.embedder = embedder
#         self.clusterer = clusterer
#         self.distiller = distiller

#     def run(self) -> List[DistilledNote]:
#         documents = self.source.load_documents()
#         embedded_docs = self.embedder.embed_documents(documents)
#         clusters = self.clusterer.cluster_documents(embedded_docs)
#         distilled_notes = self.distiller.distill_knowledge(clusters)
#         return distilled_notes


# # Factory function to create components from config
# def create_pipeline_from_config(config: Dict[str, Any]) -> DistillationPipeline:
#     source_config = config.get("source", {})
#     source = ObsidianSource(source_config.get("path"), source_config.get("config"))

#     # Create other components similarly...

#     return DistillationPipeline(source, embedder, clusterer, distiller)
