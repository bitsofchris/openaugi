# knowledge_store.py
from typing import List, Dict, Any, Optional
from llama_index.core.schema import Document
import lancedb
import os
import uuid
import json
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

    def save_raw_document(self, document: Document) -> str:
        """
        Save a single raw document with embedding.
        Returns the document ID.
        """
        # Ensure document has an embedding
        if "embedding" not in document.metadata:
            raise ValueError("Document must have an embedding before saving to LanceDB")

        # Use document ID directly
        doc_id = document.doc_id
        filepath = document.metadata.get("file_path", "")

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
                pa.field("processed_time", pa.timestamp('s')),
                pa.field("is_processed", pa.bool_()),
                pa.field("metadata_json", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), vector_dim)),
            ])

            table = self.db.create_table(self.raw_docs_table, schema=raw_schema)
        else:
            table = self.db.open_table(self.raw_docs_table)

        # Check if document exists by ID
        existing = table.search().where(f"id = '{doc_id}'").limit(1).to_pandas()

        if len(existing) == 0:
            # Document is new
            item = {
                "id": doc_id,
                "text": document.text,
                "filepath": filepath,
                "modified_time": datetime.now().replace(microsecond=0),
                "processed_time": datetime.now().replace(microsecond=0),
                "is_processed": False,
                "metadata_json": json.dumps({k: str(v) for k, v in document.metadata.items() if k != "embedding"}),
                "vector": document.metadata["embedding"]
            }

            # Add the new document
            table.add([item])
            return doc_id
        else:
            # Document already exists - no need to update since document ID
            # already encodes content hash in the source
            return doc_id

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
        row["is_processed"] = True

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

    def has_processed_document(self, raw_doc_id: str) -> bool:
        """
        Check if a raw document has been processed into atomic notes.
        """
        if self.raw_docs_table not in self.db.table_names():
            return False

        table = self.db.open_table(self.raw_docs_table)
        results = table.search().where(f"id = '{raw_doc_id}' AND is_processed = true").limit(1).to_arrow()

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

    def get_new_or_changed_documents(self, documents: List[Document]) -> List[Document]:
        """
        Compare documents against database to identify new or changed documents.

        Args:
            documents: List of documents to check

        Returns:
            List of documents that are new or have changed content
        """
        if self.raw_docs_table not in self.db.table_names():
            # All documents are new if table doesn't exist
            return documents

        new_or_changed = []
        table = self.db.open_table(self.raw_docs_table)

        for doc in documents:
            filepath = doc.metadata.get("file_path", "")
            content_hash = self._generate_hash(doc.text)

            # Check if document exists with same path and hash
            results = table.search().where(f"filepath = '{filepath}' AND content_hash = '{content_hash}'") \
                .limit(1).to_arrow()

            if len(results) == 0:
                # Document is new or content has changed
                new_or_changed.append(doc)

        return new_or_changed

    def get_new_documents(self, documents):
        """
        Filter list to only documents not already in the database.

        Args:
            documents: List of documents with IDs properly set

        Returns:
            List of documents not in the database
        """
        if self.raw_docs_table not in self.db.table_names():
            return documents

        new_docs = []
        for doc in documents:
            if self.get_raw_document(doc.doc_id) is None:
                new_docs.append(doc)

        return new_docs

    def get_unprocessed_documents(self) -> List[Document]:
        """
        Get all raw documents that haven't been processed into atomic notes.

        Returns:
            List of Documents that need atomic note extraction
        """
        if self.raw_docs_table not in self.db.table_names():
            return []

        table = self.db.open_table(self.raw_docs_table)

        # Direct query for unprocessed documents
        unprocessed_rows = table.search().where("is_processed = false").to_pandas()

        # Convert rows to Document objects
        unprocessed = []
        for _, row in unprocessed_rows.iterrows():
            try:
                metadata = json.loads(row.get('metadata_json', '{}'))
            except json.JSONDecodeError:
                metadata = {}

            if 'vector' in row and row['vector'] is not None:
                metadata['embedding'] = row['vector']

            doc = Document(
                text=row.get('text', ''),
                metadata=metadata,
                doc_id=row.get('id', '')
            )
            unprocessed.append(doc)

        return unprocessed

    def save_cluster_assignments(self, cluster_assignments: Dict[str, int]):
        """
        Save cluster assignments for atomic notes.

        Args:
            cluster_assignments: Dictionary mapping note_id to cluster_id
        """
        import pyarrow as pa
        import pandas as pd

        # Create or get cluster assignments table
        table_name = "cluster_assignments"
        if table_name not in self.db.table_names():
            schema = pa.schema([
                pa.field("note_id", pa.string()),
                pa.field("cluster_id", pa.int32()),
                pa.field("timestamp", pa.timestamp('s'))
            ])
            table = self.db.create_table(table_name, schema=schema)
        else:
            table = self.db.open_table(table_name)

        # Convert assignments to dataframe
        from datetime import datetime
        now = datetime.now().replace(microsecond=0)

        data = []
        for note_id, cluster_id in cluster_assignments.items():
            data.append({
                "note_id": note_id,
                "cluster_id": cluster_id,
                "timestamp": now
            })

        # Add to table
        df = pd.DataFrame(data)
        table.add(df)

    def get_notes_by_cluster(self, cluster_id: int) -> List[Document]:
        """
        Get all atomic notes belonging to a specific cluster.

        Args:
            cluster_id: Cluster ID to retrieve

        Returns:
            List of Document objects in the cluster
        """
        # Check if cluster assignments table exists
        table_name = "cluster_assignments"
        if table_name not in self.db.table_names():
            return []

        # Get note IDs for this cluster
        assignments_table = self.db.open_table(table_name)
        results = assignments_table.search().where(f"cluster_id = {cluster_id}").to_pandas()

        if len(results) == 0:
            return []

        # Get notes
        note_ids = results["note_id"].tolist()
        notes = []
        for note_id in note_ids:
            note = self.get_atomic_note(note_id)
            if note:
                notes.append(note)

        return notes

    def get_all_clusters(self) -> Dict[int, List[str]]:
        """
        Get all clusters and their note IDs.

        Returns:
            Dictionary mapping cluster_id to list of note_ids
        """
        # Check if cluster assignments table exists
        table_name = "cluster_assignments"
        if table_name not in self.db.table_names():
            return {}

        # Get all assignments
        assignments_table = self.db.open_table(table_name)
        results = assignments_table.search().to_pandas()

        if len(results) == 0:
            return {}

        # Group by cluster
        clusters = {}
        for _, row in results.iterrows():
            cluster_id = row["cluster_id"]
            note_id = row["note_id"]

            if cluster_id not in clusters:
                clusters[cluster_id] = []

            clusters[cluster_id].append(note_id)

        return clusters

    def get_all_atomic_notes(self) -> List[Document]:
        """
        Retrieve all atomic notes with their embeddings.
        Uses pagination to work around LanceDB query limitations.

        Returns:
            List of Document objects representing atomic notes
        """
        if self.atomic_notes_table not in self.db.table_names():
            print("No atomic notes found in database")
            return []

        table = self.db.open_table(self.atomic_notes_table)

        # Use pagination to get all notes
        batch_size = 1000
        offset = 0
        all_documents = []

        while True:
            # Get a batch of records
            results = table.search().limit(batch_size).offset(offset).to_pandas()

            # If no results returned, we've reached the end
            if len(results) == 0:
                break

            # Convert rows to Documents
            for _, row in results.iterrows():
                metadata = {}

                # Try to parse metadata_json if it exists
                if "metadata_json" in row and row["metadata_json"]:
                    try:
                        import json
                        metadata = json.loads(row["metadata_json"])
                    except json.JSONDecodeError:
                        print(f"Error decoding metadata for note {row.get('id', 'unknown')}")

                # Add other metadata fields
                metadata["idea_title"] = row.get("idea_title", "")
                metadata["links"] = row.get("links", []) if "links" in row else []
                metadata["source_doc_ids"] = row.get("source_doc_ids", []) if "source_doc_ids" in row else []

                # Add embedding if available
                if "vector" in row and row["vector"] is not None:
                    metadata["embedding"] = row["vector"]

                doc = Document(
                    text=row.get("text", ""),
                    metadata=metadata,
                    doc_id=row.get("id", "")
                )
                all_documents.append(doc)

            # Increase offset for next batch
            offset += len(results)

            # Log progress
            print(f"Loaded {len(all_documents)} atomic notes so far...")

        print(f"Retrieved {len(all_documents)} atomic notes total")
        return all_documents

    def get_already_distilled_note_ids(self) -> set:
        """
        Get IDs of atomic notes that have already been used in distilled notes.

        Returns:
            Set of atomic note IDs that have been used in distillation
        """
        if self.clean_notes_table not in self.db.table_names():
            return set()

        table = self.db.open_table(self.clean_notes_table)
        # TODO - standardize how we check ids in table -> use a KV?
        results = table.search().limit(10000).to_pandas()

        already_distilled = set()
        for _, row in results.iterrows():
            if "atomic_note_ids" in row:
                # Handle the case where atomic_note_ids is an array/list
                note_ids = row["atomic_note_ids"]
                if isinstance(note_ids, list) and len(note_ids) > 0:
                    already_distilled.update(note_ids)
                elif hasattr(note_ids, '__iter__') and not isinstance(note_ids, str):
                    # For pandas/numpy array types
                    already_distilled.update([str(id) for id in note_ids if id])

        return already_distilled

    def get_all_clean_notes(self) -> List[Document]:
        """
        Retrieve all clean/distilled notes with their embeddings.

        Returns:
            List of Document objects representing clean/distilled notes
        """
        if self.clean_notes_table not in self.db.table_names():
            return []

        table = self.db.open_table(self.clean_notes_table)

        # Get all data
        try:
            results = table.search().to_pandas()
        except Exception as e:
            print(f"Error retrieving clean notes: {e}")
            return []

        # Convert rows to Documents
        all_documents = []
        for _, row in results.iterrows():
            metadata = {}

            # Try to parse metadata_json if it exists
            if "metadata_json" in row and row["metadata_json"]:
                try:
                    import json
                    metadata = json.loads(row["metadata_json"])
                except json.JSONDecodeError:
                    print(f"Error decoding metadata for note {row.get('id', 'unknown')}")

            # Add specific fields
            metadata["title"] = row.get("title", "")
            metadata["is_clean_note"] = True

            # Add source note IDs
            if "atomic_note_ids" in row:
                metadata["atomic_note_ids"] = row["atomic_note_ids"]

            # Add sibling notes if available
            if "sibling_notes" in row:
                metadata["sibling_notes"] = row["sibling_notes"]

            # Add embedding if available
            if "vector" in row and row["vector"] is not None:
                metadata["embedding"] = row["vector"]

            doc = Document(
                text=row.get("text", ""),
                metadata=metadata,
                doc_id=row.get("id", "")
            )
            all_documents.append(doc)

        print(f"Retrieved {len(all_documents)} clean/distilled notes")
        return all_documents
