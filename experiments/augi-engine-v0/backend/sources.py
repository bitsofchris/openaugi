from typing import Dict, Any, List
from llama_index.core.schema import Document
from llama_index.readers.obsidian import ObsidianReader
from interfaces import DocumentSource
import hashlib
import json
from datetime import datetime


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
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

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


class ClaudeAISource(DocumentSource):
    def __init__(self, path: str, config: Dict[str, Any] = None):
        """
        Initialize Claude AI chat source.

        Args:
            path: Path to the Claude AI conversations JSON file
            config: Optional configuration dictionary
        """
        self.path = path
        self.config = config or {}

    def parse_timestamp_to_int(self, timestamp_str: str) -> int:
        """Convert ISO timestamp to integer for deterministic doc IDs."""
        if not timestamp_str:
            return 0
        try:
            dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            return int(dt.timestamp() * 1000000)
        except (ValueError, TypeError):
            return 0

    def extract_text_content(self, content) -> str:
        """
        Extract text content from various content formats.

        Args:
            content: Content field from chat message (list, str, or dict)

        Returns:
            str: Extracted text content
        """
        text_content = ""

        if isinstance(content, list):
            text_parts = []
            for content_item in content:
                if isinstance(content_item, dict) and "text" in content_item:
                    text_parts.append(content_item["text"])
            text_content = "\n".join(text_parts)
        elif isinstance(content, str):
            text_content = content
        elif isinstance(content, dict) and "text" in content:
            text_content = content["text"]

        return text_content

    def generate_unique_doc_id(
        self, thread_id: str, turn_idx: int, timestamp_str: str
    ) -> str:
        """
        Generate a deterministic document ID for a chat message.

        Args:
            thread_id: UUID of the conversation thread
            turn_idx: Index of the message within the conversation
            timestamp_str: ISO timestamp string

        Returns:
            str: Deterministic document ID
        """
        timestamp_int = self.parse_timestamp_to_int(timestamp_str)
        return f"{thread_id}-{turn_idx:03d}-{timestamp_int}"

    def load_documents(self) -> List[Document]:
        """
        Load Claude AI chat documents from JSON file.
        Creates one document per chat message.

        Returns:
            List[Document]: List of LlamaIndex Document objects
        """
        try:
            with open(self.path, "r", encoding="utf-8") as file:
                conversations_data = json.load(file)
        except FileNotFoundError:
            print(f"Error: Could not find file at {self.path}")
            return []
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in file {self.path}: {e}")
            return []

        docs = []

        for conversation in conversations_data:
            thread_id = conversation.get("uuid", "unknown")
            conversation_name = conversation.get("name", "Untitled Conversation")
            conversation_created = conversation.get("created_at")

            # Process each message as a separate document
            for turn_idx, message in enumerate(conversation.get("chat_messages", [])):
                # Extract text content from message
                content = message.get("content", [])
                text_content = self.extract_text_content(content)

                # Only create document if there's actual text content
                if text_content.strip():
                    # Create deterministic ID
                    created_at = message.get("created_at", "")
                    doc_id = self.generate_unique_doc_id(
                        thread_id, turn_idx, created_at
                    )

                    # Create document
                    doc = Document(
                        text=text_content,
                        metadata={
                            "thread": thread_id,
                            "thread_name": conversation_name,
                            "role": message.get(
                                "sender", "unknown"
                            ),  # "human" / "assistant"
                            "turn_index": turn_idx,
                            "conversation_created": conversation_created,
                            "message_created": message.get("created_at"),
                            "source": "claude_ai_chats",
                            "is_raw_document": True,
                            "file_path": self.path,
                        },
                    )
                    doc.doc_id = doc_id
                    docs.append(doc)

        print(f"Created {len(docs)} documents from Claude AI chats")

        # Print statistics
        if docs:
            role_counts = {}
            for doc in docs:
                role = doc.metadata.get("role", "unknown")
                role_counts[role] = role_counts.get(role, 0) + 1

            print(f"Messages by role: {role_counts}")

            # Count unique conversations
            unique_threads = len(set(doc.metadata["thread"] for doc in docs))
            print(f"From {unique_threads} unique conversations")

        return docs
