# storage.py
from typing import List, Dict, Any
from llama_index.core.schema import Document
import lancedb
import os
import uuid
import time


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

    def has_processed_document(self, doc_id: str) -> bool:
        """
        Check if a document has already been processed into atomic ideas.
        Args:
            doc_id: ID of the source document
        Returns:
            True if document has been processed, False otherwise
        """
        if self.table_name not in self.db.table_names():
            return False

        table = self.db.open_table(self.table_name)

        # Fixed: Use search() or where() method for filtering instead of to_arrow() with filter parameter
        results = table.search().where(f"source_doc_id = '{doc_id}'").to_arrow()

        return len(results) > 0

    def get_atomic_ideas_for_doc(self, doc_id: str) -> List[Dict[str, Any]]:
        """
        Get all atomic ideas derived from a specific document.

        Args:
            doc_id: ID of the source document

        Returns:
            List of atomic ideas with metadata
        """
        if self.table_name not in self.db.table_names():
            return []

        table = self.db.open_table(self.table_name)
        results = table.to_pandas(filter=f"source_doc_id = '{doc_id}'")
        return results.to_dict(orient="records")

    def get_all_atomic_ideas_as_documents(self) -> List[Document]:
        """
        Get all atomic ideas from the database as Document objects.

        Returns:
            List of Document objects representing atomic ideas
        """
        from llama_index.core.schema import Document

        if self.table_name not in self.db.table_names():
            return []

        table = self.db.open_table(self.table_name)
        results = table.to_pandas()

        documents = []
        for _, row in results.iterrows():
            metadata = {
                "source_doc_id": row.get("source_doc_id", ""),
                "source_doc_title": row.get("source_doc_title", ""),
                "source_doc_path": row.get("source_doc_path", ""),
                "idea_title": row.get("idea_title", ""),
                "links": row.get("links", []),
                "is_atomic_idea": True
            }
            doc = Document(
                text=row.get("text", ""),
                metadata=metadata,
                doc_id=row.get("id", "")
            )
            documents.append(doc)

        return documents

    def save_documents_with_embeddings(self, documents: List[Document]) -> None:
        """
        Save documents (atomic ideas or full documents) with their embeddings to LanceDB.
        Args:
            documents: List of Document objects with embeddings in metadata
        """
        if not documents:
            print("No documents to save")
            return

        # Prepare data for LanceDB
        data = []
        for doc in documents:
            # Check if document has embedding
            if "embedding" not in doc.metadata:
                print(f"Warning: Document {doc.doc_id} has no embedding, skipping")
                continue

            # Get embedding vector
            embedding_vector = doc.metadata["embedding"]

            # Create item for LanceDB
            item = {
                "id": doc.doc_id,
                "text": doc.text,
                "source_doc_id": doc.metadata.get("source_doc_id", doc.doc_id),
                "source_doc_title": doc.metadata.get("source_doc_title", ""),
                "source_doc_path": doc.metadata.get("source_doc_path", ""),
                "idea_title": doc.metadata.get("idea_title", ""),
                "links": doc.metadata.get("links", []),
                "is_atomic_idea": doc.metadata.get("is_atomic_idea", False),
                "vector": embedding_vector
            }
            data.append(item)

        import pyarrow as pa
        vector_dim = len(data[0]["vector"])
        schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("source_doc_id", pa.string()),
            pa.field("source_doc_title", pa.string()),
            pa.field("source_doc_path", pa.string()),
            pa.field("idea_title", pa.string()),
            pa.field("links", pa.list_(pa.string())),
            pa.field("is_atomic_idea", pa.bool_()),
            pa.field("vector", pa.list_(pa.float32(), vector_dim)),
        ])

        # Check if table exists
        if self.table_name in self.db.table_names():
            # Try to open the table
            table = self.db.open_table(self.table_name)

            try:
                # Check if the vector column exists
                test_query = table.search().limit(1).to_arrow()
                if 'vector' not in test_query.column_names:
                    print(f"Table {self.table_name} exists but is missing vector column. Migrating data...")

                    # Backup existing data
                    existing_data = table.search().to_arrow().to_pylist()
                    print(f"Backing up {len(existing_data)} existing records")

                    # Create new table with correct schema
                    backup_table_name = f"{self.table_name}_backup_{int(time.time())}"
                    self.db.create_table(backup_table_name, data=existing_data)
                    print(f"Created backup in table {backup_table_name}")

                    # Drop existing table
                    self.db.drop_table(self.table_name)

                    # Create new table with correct schema
                    table = self.db.create_table(self.table_name, schema=schema)

                    # Restore existing data with default vector values
                    for item in existing_data:
                        # Add a default vector (zeros) for existing items
                        if "vector" not in item:
                            item["vector"] = [0.0] * vector_dim

                    # Add the restored data
                    if existing_data:
                        table.add(data=existing_data)
                        print(f"Restored {len(existing_data)} records with default vectors")
            except Exception as e:
                print(f"Error checking table schema: {str(e)}")
                # Continue with the existing table
        else:
            # Create new table
            table = self.db.create_table(self.table_name, schema=schema)

        # Add the new data
        table.add(data=data, mode="overwrite")
        print(f"Added {len(data)} documents with embeddings")

    def get_documents_without_embeddings(self) -> List[Document]:
        """
        Get all documents from LanceDB that don't have embeddings yet.

        Returns:
            List of Document objects without embeddings
        """
        if self.table_name not in self.db.table_names():
            return []

        table = self.db.open_table(self.table_name)

        # Try to identify records without embeddings
        # Note: This approach depends on how LanceDB handles missing vector values
        # Might need adjustment based on actual LanceDB behavior
        try:
            results = table.to_pandas()
            # Filter for rows where vector is null or empty
            docs_without_embeddings = []

            for idx, row in results.iterrows():
                if "vector" not in row or row["vector"] is None or len(row["vector"]) == 0:
                    metadata = {
                        "source_doc_id": row.get("source_doc_id", ""),
                        "source_doc_title": row.get("source_doc_title", ""),
                        "source_doc_path": row.get("source_doc_path", ""),
                        "idea_title": row.get("idea_title", ""),
                        "links": row.get("links", []),
                        "is_atomic_idea": row.get("is_atomic_idea", False)
                    }
                    doc = Document(
                        text=row.get("text", ""),
                        metadata=metadata,
                        doc_id=row.get("id", "")
                    )
                    docs_without_embeddings.append(doc)

            return docs_without_embeddings
        except Exception as e:
            print(f"Error getting documents without embeddings: {e}")
            return []
