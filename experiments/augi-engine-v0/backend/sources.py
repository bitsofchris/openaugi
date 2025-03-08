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
        file_path = f"{doc.metadata["folder_path"]}/{doc.metadata["file_name"]}"
        content_hash = hashlib.md5(doc.text.encode()).hexdigest()[:10]
        return f"{file_path}_{content_hash}"

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
        print(f"Found {len(documents)} documents in {self.path}")
        return documents
