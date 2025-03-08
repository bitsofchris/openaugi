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

        Args:
            documents: List of Document objects to embed

        Returns:
            Same list of Documents with embeddings added to metadata
        """
        if not documents:
            return documents

        # Extract text from documents
        texts = [doc.text for doc in documents]

        # Generate embeddings in batches
        embeddings = self.embed_model.get_text_embedding_batch(texts)

        # Add embeddings to document metadata
        for i, doc in enumerate(documents):
            doc.metadata["embedding"] = embeddings[i]

        print(f"Generated embeddings for {len(documents)} documents")
        return documents