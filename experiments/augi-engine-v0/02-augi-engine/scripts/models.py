from typing import List, Dict, Any
from llama_index.core.schema import Document


class DistilledNote(Document):
    """
    A Document subclass representing a distilled atomic note.
    Adds tracking for parent documents and sibling relationships.
    """

    def __init__(
        self,
        text: str,
        parent_documents: List[str] = None,  # IDs of parent documents
        sibling_notes: List[str] = None,  # IDs of related/sibling notes
        is_user_edited: bool = False,
        **kwargs
    ):
        # Initialize the base Document class
        super().__init__(text=text, **kwargs)

        # Add our custom properties
        self.parent_documents = parent_documents or []
        self.sibling_notes = sibling_notes or []
        self.is_user_edited = is_user_edited

    @classmethod
    def from_document(
        cls,
        doc: Document,
        parent_documents: List[str] = None,
        sibling_notes: List[str] = None,
    ) -> "DistilledNote":
        """
        Create a DistilledNote from an existing Document.
        """
        # Copy all the existing document properties
        metadata = doc.metadata.copy() if doc.metadata else {}

        return cls(
            text=doc.text,
            metadata=metadata,
            doc_id=doc.doc_id,
            embedding=doc.embedding,
            parent_documents=parent_documents,
            sibling_notes=sibling_notes,
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary including custom properties.
        """
        # Get the base document dictionary
        doc_dict = super().to_dict()

        # Add our custom properties
        doc_dict["parent_documents"] = self.parent_documents
        doc_dict["sibling_notes"] = self.sibling_notes
        doc_dict["is_user_edited"] = self.is_user_edited

        return doc_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DistilledNote":
        """
        Create a DistilledNote from a dictionary.
        """
        # Extract our custom properties
        parent_documents = data.pop("parent_documents", [])
        sibling_notes = data.pop("sibling_notes", [])
        is_user_edited = data.pop("is_user_edited", False)

        # Create a base Document first
        doc = Document.from_dict(data)

        # Then convert to a DistilledNote
        return cls.from_document(
            doc=doc, parent_documents=parent_documents, sibling_notes=sibling_notes
        )
