# storage.py
from typing import List, Dict, Any
from llama_index.core.schema import Document
import lancedb
import os
import uuid


class AtomicIdeaStore:
    def __init__(self, db_path: str, table_name: str = "atomic_ideas"):
        """
        Initialize the LanceDB storage for atomic ideas.

        Args:
            db_path: Path to the LanceDB database
            table_name: Name of the table to store atomic ideas
        """
        self.db_path = db_path
        self.table_name = table_name
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db = lancedb.connect(db_path)

    def _create_table_if_not_exists(self, schema):
        """Create the table if it doesn't exist yet."""
        if self.table_name not in self.db.table_names():
            self.db.create_table(self.table_name, schema=schema)

    def save_atomic_ideas(
        self, documents: List[Document], embeddings: List[List[float]] = None
    ) -> None:
        """
        Save atomic ideas to LanceDB, with or without embeddings.

        Args:
            documents: List of Document objects representing atomic ideas
            embeddings: Optional list of embedding vectors for the documents
        """
        if not documents:
            print("No documents to save")
            return

        # Prepare data for LanceDB
        data = []
        for i, doc in enumerate(documents):
            # Generate a unique ID if not present
            doc_id = doc.doc_id or str(uuid.uuid4())

            item = {
                "id": doc_id,
                "text": doc.text,
                "source_doc_id": doc.metadata.get("source_doc_id", ""),
                "source_doc_title": doc.metadata.get("source_doc_title", ""),
                "source_doc_path": doc.metadata.get("source_doc_path", ""),
                "idea_title": doc.metadata.get("idea_title", ""),
                "links": doc.metadata.get("links", []),
                "is_atomic_idea": doc.metadata.get("is_atomic_idea", True),
            }

            # Add embedding if available
            if embeddings and i < len(embeddings):
                item["vector"] = embeddings[i]

            data.append(item)

        # Create or get the table
        if self.table_name not in self.db.table_names():
            # If we have embeddings, include them in the schema
            if embeddings and len(embeddings) > 0:
                import pyarrow as pa

                vector_dim = len(embeddings[0])
                schema = pa.schema(
                    [
                        pa.field("id", pa.string()),
                        pa.field("text", pa.string()),
                        pa.field("source_doc_id", pa.string()),
                        pa.field("source_doc_title", pa.string()),
                        pa.field("source_doc_path", pa.string()),
                        pa.field("idea_title", pa.string()),
                        pa.field("links", pa.list_(pa.string())),
                        pa.field("is_atomic_idea", pa.bool_()),
                        pa.field("vector", pa.list_(pa.float32(), vector_dim)),
                    ]
                )
                table = self.db.create_table(self.table_name, schema=schema)
            else:
                table = self.db.create_table(self.table_name, data=data)
        else:
            table = self.db.open_table(self.table_name)

        # Add the data to the table
        table.add(data)
        print(
            f"Saved {len(documents)} atomic ideas to LanceDB table '{self.table_name}'"
        )

    def query_similar_ideas(
        self, query_embedding: List[float], limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Query LanceDB for similar ideas using vector similarity.

        Args:
            query_embedding: The embedding vector to search with
            limit: Maximum number of results to return

        Returns:
            List of matching documents with metadata
        """
        table = self.db.open_table(self.table_name)
        results = table.search(query_embedding).limit(limit).to_pandas()

        # Convert to dictionary format
        return results.to_dict(orient="records")

    def get_all_atomic_ideas(self) -> List[Dict[str, Any]]:
        """
        Retrieve all atomic ideas from the database.

        Returns:
            List of all atomic ideas with metadata
        """
        if self.table_name not in self.db.table_names():
            return []

        table = self.db.open_table(self.table_name)
        results = table.to_pandas()
        return results.to_dict(orient="records")
