# knowledge_store.py
from typing import List, Dict, Any, Optional
from llama_index.core.schema import Document
import lancedb
import os
import uuid
import json
import hashlib
from datetime import datetime


class KnowledgeStore:
    """
    Unified storage system for AugI Engine that handles all document levels:
    - Raw Documents: Original unprocessed files
    - Atomic Notes: Extracted atomic ideas from raw documents
    - Clean/Distilled Notes: Clustered and refined concepts
    """

    def __init__(self, db_path: str):
        """
        Initialize the LanceDB storage for all document types.

        Args:
            db_path: Path to the LanceDB database
        """
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db = lancedb.connect(db_path)

        # Table names for each document level
        self.raw_docs_table = "raw_documents"
        self.atomic_notes_table = "atomic_notes"
        self.clean_notes_table = "clean_notes"

        # Initialize tables if they don't exist
        self._initialize_tables()

    def _initialize_tables(self):
        """Initialize all tables with appropriate schemas if they don't exist."""
        # Tables will be created when first documents are added
        # This allows us to dynamically set vector dimensions based on embeddings
        pass

    def _generate_hash(self, content: str) -> str:
        """Generate a hash of the content for change detection."""
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    # Raw Documents Methods

    def save_raw_document(self, document: Document) -> str:
        """
        Save a single raw document with embedding.
        Returns the document ID.
        """
        # Ensure document has an embedding
        if "embedding" not in document.metadata:
            raise ValueError("Document must have an embedding before saving to LanceDB")

        content_hash = self._generate_hash(document.text)
        filepath = document.metadata.get("file_path", "")
        doc_id = document.doc_id or f"{filepath}_{content_hash}"

        # Create or get table
        if self.raw_docs_table not in self.db.table_names():
            import pyarrow as pa

            # Get embedding dimension from the document
            vector_dim = len(document.metadata["embedding"])

            raw_schema = pa.schema([
                pa.field("id", pa.string()),
                pa.field("text", pa.string()),
                pa.field("filepath", pa.string()),
                pa.field("modified_time", pa.timestamp('s')),
                pa.field("content_hash", pa.string()),
                pa.field("processed_time", pa.timestamp('s')),
                pa.field("metadata_json", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), vector_dim)),
            ])

            table = self.db.create_table(self.raw_docs_table, schema=raw_schema)
        else:
            table = self.db.open_table(self.raw_docs_table)

        # Check if document exists and has changed
        existing = table.search().where(f"filepath = '{filepath}'").limit(1).to_pandas()

        if len(existing) == 0 or existing.iloc[0]["content_hash"] != content_hash:
            # New or changed document
            item = {
                "id": doc_id,
                "text": document.text,
                "filepath": filepath,
                "modified_time": datetime.now().replace(microsecond=0),
                "content_hash": content_hash,
                "processed_time": datetime.now().replace(microsecond=0),
                "metadata_json": json.dumps({k: str(v) for k, v in document.metadata.items() if k != "embedding"}),
                "vector": document.metadata["embedding"]
            }

            # Delete if exists (for update)
            if len(existing) > 0:
                table.delete(f"filepath = '{filepath}'")

            # Add the new/updated document
            table.add([item])

            return doc_id

        # No change, return existing ID
        return existing.iloc[0]["id"]

    def save_raw_documents(self, documents: List[Document]) -> List[str]:
        """
        Save multiple raw documents, returning list of document IDs.
        All documents must have embeddings.
        """
        return [self.save_raw_document(doc) for doc in documents]

    def get_raw_document(self, doc_id: str) -> Optional[Document]:
        """
        Retrieve a specific raw document by ID.
        """
        if self.raw_docs_table not in self.db.table_names():
            return None

        table = self.db.open_table(self.raw_docs_table)
        results = table.search().where(f"id = '{doc_id}'").limit(1).to_pandas()

        if len(results) == 0:
            return None

        row = results.iloc[0]
        metadata = row.get("metadata", {})
        if "vector" in row and row["vector"] is not None:
            metadata["embedding"] = row["vector"]

        doc = Document(
            text=row.get("text", ""),
            metadata=metadata,
            doc_id=row.get("id", "")
        )

        return doc

    def mark_raw_document_processed(self, doc_id: str) -> None:
        """
        Mark a raw document as processed, updating the processed_time.
        """
        if self.raw_docs_table not in self.db.table_names():
            return

        table = self.db.open_table(self.raw_docs_table)
        results = table.search().where(f"id = '{doc_id}'").limit(1).to_pandas()

        if len(results) == 0:
            return

        # Update the processed time
        row = results.iloc[0].to_dict()
        row["processed_time"] = datetime.now().replace(microsecond=0)

        # Delete and re-add (update)
        table.delete(f"id = '{doc_id}'")
        table.add([row])

    # Atomic Notes Methods

    def save_atomic_note(self, note: Document) -> str:
        """
        Save a single atomic note with embedding, returning the note ID.
        """
        # Ensure note has an embedding
        if "embedding" not in note.metadata:
            raise ValueError("Atomic note must have an embedding before saving to LanceDB")

        # Create or get table
        if self.atomic_notes_table not in self.db.table_names():
            import pyarrow as pa

            # Get embedding dimension from the note
            vector_dim = len(note.metadata["embedding"])

            atomic_schema = pa.schema([
                pa.field("id", pa.string()),
                pa.field("text", pa.string()),
                pa.field("idea_title", pa.string()),
                pa.field("source_doc_ids", pa.list_(pa.string())),  # Links to raw documents
                pa.field("created_time", pa.timestamp('s')),
                pa.field("metadata_json", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), vector_dim)),
                pa.field("links", pa.list_(pa.string())),  # Related concepts
            ])

            table = self.db.create_table(self.atomic_notes_table, schema=atomic_schema)
        else:
            table = self.db.open_table(self.atomic_notes_table)

        # Ensure the note has an ID
        note_id = note.doc_id or str(uuid.uuid4())

        # Get source document ID(s)
        source_doc_id = note.metadata.get("source_doc_id", "")
        source_doc_ids = [source_doc_id] if source_doc_id else []

        item = {
            "id": note_id,
            "text": note.text,
            "idea_title": note.metadata.get("idea_title", ""),
            "source_doc_ids": source_doc_ids,
            "created_time": datetime.now().replace(microsecond=0),
            "metadata_json": json.dumps({k: str(v) for k, v in note.metadata.items() if k != "embedding"}),
            "vector": note.metadata["embedding"],
            "links": note.metadata.get("links", [])
        }

        table.add([item])
        return note_id

    def save_atomic_notes(self, notes: List[Document]) -> List[str]:
        """
        Save multiple atomic notes with embeddings, returning list of note IDs.
        """
        return [self.save_atomic_note(note) for note in notes]

    def get_atomic_note(self, note_id: str) -> Optional[Document]:
        """
        Retrieve a specific atomic note by ID.
        """
        if self.atomic_notes_table not in self.db.table_names():
            return None

        table = self.db.open_table(self.atomic_notes_table)
        results = table.search().where(f"id = '{note_id}'").limit(1).to_pandas()

        if len(results) == 0:
            return None

        row = results.iloc[0]
        metadata = row.get("metadata", {})
        metadata["idea_title"] = row.get("idea_title", "")
        metadata["links"] = row.get("links", [])
        metadata["source_doc_ids"] = row.get("source_doc_ids", [])

        if "vector" in row and row["vector"] is not None:
            metadata["embedding"] = row["vector"]

        doc = Document(
            text=row.get("text", ""),
            metadata=metadata,
            doc_id=row.get("id", "")
        )

        return doc

    def get_atomic_notes_for_raw_document(self, raw_doc_id: str) -> List[Document]:
        """
        Get all atomic notes derived from a specific raw document.
        """
        if self.atomic_notes_table not in self.db.table_names():
            return []

        table = self.db.open_table(self.atomic_notes_table)
        results = table.search().where(f"'{raw_doc_id}' IN source_doc_ids").to_pandas()

        notes = []
        for _, row in results.iterrows():
            metadata = row.get("metadata", {})
            metadata["idea_title"] = row.get("idea_title", "")
            metadata["links"] = row.get("links", [])
            metadata["source_doc_ids"] = row.get("source_doc_ids", [])

            if "vector" in row and row["vector"] is not None:
                metadata["embedding"] = row["vector"]

            doc = Document(
                text=row.get("text", ""),
                metadata=metadata,
                doc_id=row.get("id", "")
            )

            notes.append(doc)

        return notes

    # Clean Notes Methods

    def save_clean_note(self, note: Document, atomic_note_ids: List[str]) -> str:
        """
        Save a single clean/distilled note with embedding, returning the note ID.
        """
        # Ensure note has an embedding
        if "embedding" not in note.metadata:
            raise ValueError("Clean note must have an embedding before saving to LanceDB")

        # Create or get table
        if self.clean_notes_table not in self.db.table_names():
            import pyarrow as pa

            # Get embedding dimension from the note
            vector_dim = len(note.metadata["embedding"])

            clean_schema = pa.schema([
                pa.field("id", pa.string()),
                pa.field("text", pa.string()),
                pa.field("title", pa.string()),
                pa.field("atomic_note_ids", pa.list_(pa.string())),  # Links to atomic notes
                pa.field("created_time", pa.timestamp('s')),
                pa.field("last_modified_time", pa.timestamp('s')),
                pa.field("is_user_edited", pa.bool_()),
                pa.field("metadata_json", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), vector_dim)),
                pa.field("sibling_notes", pa.list_(pa.string())),  # Related concepts
            ])

            table = self.db.create_table(self.clean_notes_table, schema=clean_schema)
        else:
            table = self.db.open_table(self.clean_notes_table)

        # Ensure the note has an ID
        note_id = note.doc_id or str(uuid.uuid4())

        item = {
            "id": note_id,
            "text": note.text,
            "title": note.metadata.get("title", ""),
            "atomic_note_ids": atomic_note_ids,
            "created_time": datetime.now().replace(microsecond=0),
            "last_modified_time": datetime.now().replace(microsecond=0),
            "is_user_edited": note.metadata.get("is_user_edited", False),
            "metadata_json": json.dumps({k: str(v) for k, v in note.metadata.items() if k != "embedding"}),
            "vector": note.metadata["embedding"],
            "sibling_notes": note.metadata.get("sibling_notes", [])
        }

        table.add([item])
        return note_id

    def save_clean_notes(self, notes: List[Document], atomic_note_ids_map: Dict[str, List[str]]) -> List[str]:
        """
        Save multiple clean/distilled notes with embeddings, returning list of note IDs.
        Map of note ID to atomic note IDs must be provided.
        """
        note_ids = []
        for note in notes:
            atomic_ids = atomic_note_ids_map.get(note.doc_id, [])
            note_id = self.save_clean_note(note, atomic_ids)
            note_ids.append(note_id)
        return note_ids

    def get_clean_note(self, note_id: str) -> Optional[Document]:
        """
        Retrieve a specific clean/distilled note by ID.
        """
        if self.clean_notes_table not in self.db.table_names():
            return None

        table = self.db.open_table(self.clean_notes_table)
        results = table.search().where(f"id = '{note_id}'").limit(1).to_pandas()

        if len(results) == 0:
            return None

        row = results.iloc[0]
        metadata = row.get("metadata", {})
        metadata["title"] = row.get("title", "")
        metadata["atomic_note_ids"] = row.get("atomic_note_ids", [])
        metadata["is_user_edited"] = row.get("is_user_edited", False)
        metadata["sibling_notes"] = row.get("sibling_notes", [])

        if "vector" in row and row["vector"] is not None:
            metadata["embedding"] = row["vector"]

        doc = Document(
            text=row.get("text", ""),
            metadata=metadata,
            doc_id=row.get("id", "")
        )

        return doc

    def get_clean_notes_for_atomic_note(self, atomic_note_id: str) -> List[Document]:
        """
        Get all clean/distilled notes that include a specific atomic note.
        """
        if self.clean_notes_table not in self.db.table_names():
            return []

        table = self.db.open_table(self.clean_notes_table)
        results = table.search().where(f"'{atomic_note_id}' IN atomic_note_ids").to_pandas()

        notes = []
        for _, row in results.iterrows():
            metadata = row.get("metadata", {})
            metadata["title"] = row.get("title", "")
            metadata["atomic_note_ids"] = row.get("atomic_note_ids", [])
            metadata["is_user_edited"] = row.get("is_user_edited", False)
            metadata["sibling_notes"] = row.get("sibling_notes", [])

            if "vector" in row and row["vector"] is not None:
                metadata["embedding"] = row["vector"]

            doc = Document(
                text=row.get("text", ""),
                metadata=metadata,
                doc_id=row.get("id", "")
            )

            notes.append(doc)

        return notes

    def update_clean_note(self, note: Document) -> None:
        """
        Update an existing clean/distilled note (e.g. after user edit).
        Note must have embedding.
        """
        # Ensure note has an embedding
        if "embedding" not in note.metadata:
            raise ValueError("Clean note must have an embedding before saving to LanceDB")

        if self.clean_notes_table not in self.db.table_names():
            return

        table = self.db.open_table(self.clean_notes_table)

        # Get the existing note to preserve relationships
        results = table.search().where(f"id = '{note.doc_id}'").limit(1).to_pandas()

        if len(results) == 0:
            # Note doesn't exist, nothing to update
            return

        # Preserve the atomic note IDs and other metadata
        row = results.iloc[0]
        atomic_note_ids = row.get("atomic_note_ids", [])

        # Update the note
        item = {
            "id": note.doc_id,
            "text": note.text,
            "title": note.metadata.get("title", ""),
            "atomic_note_ids": atomic_note_ids,
            "created_time": row.get("created_time", datetime.now().replace(microsecond=0)),
            "last_modified_time": datetime.now().replace(microsecond=0),
            "is_user_edited": True,
            "metadata_json": json.dumps({k: str(v) for k, v in note.metadata.items() if k != "embedding"}),
            "vector": note.metadata["embedding"],
            "sibling_notes": note.metadata.get("sibling_notes", [])
        }

        # Delete and re-add (update)
        table.delete(f"id = '{note.doc_id}'")
        table.add([item])

    # Utility Methods

    def has_processed_document(self, raw_doc_id: str) -> bool:
        """
        Check if a raw document has been processed into atomic notes.
        """
        if self.atomic_notes_table not in self.db.table_names():
            return False

        table = self.db.open_table(self.atomic_notes_table)
        results = table.search().where(f"'{raw_doc_id}' IN source_doc_ids").limit(1).to_arrow()

        return len(results) > 0

    def search_similar_documents(
        self,
        query_embedding: List[float],
        table_name: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents in the specified table using vector similarity.

        Args:
            query_embedding: The embedding vector to search with
            table_name: Which table to search (raw_documents, atomic_notes, or clean_notes)
            limit: Maximum number of results to return

        Returns:
            List of matching documents with metadata
        """
        if table_name not in self.db.table_names():
            return []

        table = self.db.open_table(table_name)
        results = table.search(query_embedding).limit(limit).to_pandas()

        return results.to_dict(orient="records")