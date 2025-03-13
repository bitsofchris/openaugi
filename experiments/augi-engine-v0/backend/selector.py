from typing import List
import numpy as np
from llama_index.core.schema import Document


class IntraClusterSimilarityFilter:
    """
    Filters documents within clusters based on cosine similarity.
    Only groups very similar documents within the same cluster.
    """

    def __init__(self, similarity_threshold: float = 0.85, min_group_size: int = 2):
        """
        Initialize the similarity filter.

        Args:
            similarity_threshold: Minimum cosine similarity to consider documents highly related
            min_group_size: Minimum number of documents in a similarity group
        """
        self.similarity_threshold = similarity_threshold
        self.min_group_size = min_group_size

    def _compute_similarity_matrix(self, documents: List[Document]) -> np.ndarray:
        """Compute cosine similarity matrix between documents."""
        # Extract embeddings
        embeddings = []
        for doc in documents:
            if "embedding" not in doc.metadata:
                raise ValueError(f"Document {doc.doc_id} missing embedding")
            embeddings.append(doc.metadata["embedding"])

        embeddings = np.array(embeddings)

        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / norms

        # Compute similarity matrix
        similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)

        return similarity_matrix

    def find_similarity_groups(self, cluster: List[Document]) -> List[List[Document]]:
        """
        Find groups of highly similar documents within a cluster.

        Args:
            cluster: A cluster of documents

        Returns:
            List of document groups (each group is a list of highly similar Documents)
        """
        if len(cluster) < self.min_group_size:
            return []

        # Compute similarity matrix
        similarity_matrix = self._compute_similarity_matrix(cluster)

        # Find groups of similar documents
        similarity_groups = []
        processed = set()

        for i in range(len(cluster)):
            if i in processed:
                continue

            # Find similar documents
            similar_indices = np.where(similarity_matrix[i] >= self.similarity_threshold)[0]

            # Filter out already processed documents
            similar_indices = [j for j in similar_indices if j not in processed]

            # Create group if it meets minimum size
            if len(similar_indices) >= self.min_group_size:
                group = [cluster[j] for j in similar_indices]
                similarity_groups.append(group)
                processed.update(similar_indices)

        return similarity_groups

    def filter_clusters(self, clusters: List[List[Document]]) -> List[List[Document]]:
        """
        Process clusters to find similarity groups within each cluster.

        Args:
            clusters: List of document clusters

        Returns:
            List of similarity groups (flat list of all groups from all clusters)
        """
        all_similarity_groups = []

        for i, cluster in enumerate(clusters):
            similarity_groups = self.find_similarity_groups(cluster)
            all_similarity_groups.extend(similarity_groups)
            print(f"Cluster {i+1}: Found {len(similarity_groups)} similarity groups")

        return all_similarity_groups
