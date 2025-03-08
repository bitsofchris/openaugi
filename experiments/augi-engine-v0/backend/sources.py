from typing import Dict, Any, List
from llama_index.core.schema import Document
from llama_index.readers.obsidian import ObsidianReader
from interfaces import DocumentSource


class ObsidianSource(DocumentSource):
    def __init__(self, path: str, config: Dict[str, Any] = None):
        self.path = path
        self.config = config or {}
        self.extract_tasks = self.config.get("extract_tasks", True)
        self.remove_tasks = self.config.get("remove_tasks_from_text", True)

    def load_documents(self) -> List[Document]:
        reader = ObsidianReader(
            self.path,
            extract_tasks=self.extract_tasks,
            remove_tasks_from_text=self.remove_tasks,
        )
        documents = reader.load_data()
        for doc in documents:
            # Use file path as a deterministic document ID
            file_path = f"{doc.metadata["folder_path"]}/{doc.metadata["file_name"]}"
            doc.doc_id = file_path
        print(f"Found {len(documents)} documents in {self.path}")
        return documents
