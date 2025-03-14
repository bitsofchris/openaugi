from typing import Dict, Any, List
from llama_index.core.schema import Document
from llama_index.readers.obsidian import ObsidianReader
from interfaces import DocumentSource
import hashlib


class ObsidianSource(DocumentSource):
    def __init__(self, path: str, config: Dict[str, Any] = None):
        self.path = path
        self.config = config or {}
        self.extract_tasks = self.config.get("extract_tasks", True)
        self.remove_tasks = self.config.get("remove_tasks_from_text", True)

    def generate_unique_doc_id(self, doc):
        """
        Generate a stable, unique document ID based on file path and content
        using SHA-256 hashing to avoid special character issues.

        Args:
            doc: Document object with metadata and text content

        Returns:
            str: A hexadecimal hash that uniquely identifies the document
        """
        file_path = f"{doc.metadata['folder_path']}/{doc.metadata['file_name']}"
        doc.metadata["file_path"] = file_path
        combined = f"{file_path}:{doc.text}"
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()

    def load_documents(self) -> List[Document]:
        reader = ObsidianReader(
            self.path,
            extract_tasks=self.extract_tasks,
            remove_tasks_from_text=self.remove_tasks,
        )
        documents = reader.load_data()
        for doc in documents:
            # Use file path as a deterministic document ID
            doc.doc_id = self.generate_unique_doc_id(doc)
            doc.metadata["is_raw_document"] = True

        print(f"Found {len(documents)} documents in {self.path}")
        return documents
