# embedder.py
from typing import List
from llama_index.core.schema import Document
from llama_index.embeddings.openai import OpenAIEmbedding
from interfaces import Embedder


class LlamaIndexEmbedder(Embedder):
    def __init__(self, model_name="text-embedding-3-small"):
        """
        Initialize the embedder using LlamaIndex's OpenAI embedding model.

        Args:
            model_name: Name of the OpenAI embedding model to use
        """
        from dotenv import load_dotenv
        load_dotenv("/Users/chris/repos/openaugi/keys.env")
        self.embed_model = OpenAIEmbedding(model_name=model_name)

    def embed_documents(self, documents: List[Document]) -> List[Document]:
        """
        Generate embeddings for a list of documents.
        Stores the embedding in each document's metadata.
        Documents larger than 8000 characters are processed in chunks.
        Failed embeddings are logged and marked with embedding=None.

        Args:
            documents: List of Document objects to embed
        Returns:
            List of Documents with embeddings added to metadata where possible
        """
        if not documents:
            return documents

        MAX_CHARS = 8000
        successful_count = 0
        failed_count = 0

        # Process each document individually for maximum resilience
        for doc in documents:
            if not isinstance(doc.text, str) or not doc.text.strip():
                # Handle invalid text
                print(f"Warning: Skipping document with invalid text: {repr(doc.text)[:100]}")
                doc.metadata["embedding"] = None
                doc.metadata["embedding_error"] = "Invalid or empty text"
                failed_count += 1
                continue

            try:
                if len(doc.text) <= MAX_CHARS:
                    # Standard document - embed directly
                    embedding = self.embed_model.get_text_embedding(doc.text)
                    doc.metadata["embedding"] = embedding
                    successful_count += 1
                else:
                    # Large document - split and process in chunks
                    embedding = self._embed_large_document(doc.text, MAX_CHARS)
                    doc.metadata["embedding"] = embedding
                    successful_count += 1
            except Exception as e:
                # Document failed to embed - log error and mark with None
                error_msg = str(e)
                doc.metadata["embedding"] = None
                doc.metadata["embedding_error"] = error_msg

                # Log only the first 100 characters of problematic documents
                doc_preview = doc.text[:100] + "..." if len(doc.text) > 100 else doc.text
                print(f"Failed to embed document: {repr(doc_preview)} - Error: {error_msg}")
                failed_count += 1

        print(f"Embedding complete: {successful_count} successful, {failed_count} failed")
        return documents

    def _embed_large_document(self, text: str, max_chars: int) -> List[float]:
        """
        Embed a large document by breaking it into chunks and averaging the embeddings.

        Args:
            text: The document text to embed
            max_chars: Maximum number of characters per chunk

        Returns:
            List[float]: The averaged embedding vector

        Raises:
            ValueError: If all chunks fail to embed
        """
        # Split text into chunks of max_chars
        chunks = [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

        # Generate embeddings for each chunk
        chunk_embeddings = []
        successful_chunks = 0
        failed_chunks = 0

        for i, chunk in enumerate(chunks):
            try:
                embedding = self.embed_model.get_text_embedding(chunk)
                chunk_embeddings.append(embedding)
                successful_chunks += 1
            except Exception as e:
                failed_chunks += 1
                # Only print an error message for the first few failures to avoid log spam
                if failed_chunks <= 3:
                    print(f"Warning: Failed to embed chunk {i} of {len(chunks)}: {str(e)}")

        if not chunk_embeddings:
            raise ValueError(f"Failed to generate embeddings for any chunk (all {len(chunks)} chunks failed)")

        if failed_chunks > 0:
            print(f"Note: Document partially embedded ({successful_chunks} of {len(chunks)} chunks successful)")

        # Average the embeddings (simple pooling strategy)
        avg_embedding = [sum(values) / len(values) for values in zip(*chunk_embeddings)]

        return avg_embedding
