from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
import numpy as np
import umap
import hdbscan
from llama_index.core.schema import Document


class Clusterer(ABC):
    @abstractmethod
    def cluster_documents(self, documents: List[Document]) -> List[List[Document]]:
        """
        Cluster a list of documents into groups of related documents.

        Args:
            documents: List of documents to cluster

        Returns:
            List of document clusters (each cluster is a list of Documents)
        """
        pass


class UMAPHDBSCANClusterer(Clusterer):
    """
    Clusters documents using UMAP dimensionality reduction and HDBSCAN clustering.
    Automatically adjusts parameters based on dataset size.
    """
    def __init__(self,
                 umap_n_components: int = 30,
                 umap_n_neighbors: int = 15,
                 umap_min_dist: float = 0.05,
                 hdbscan_min_cluster_size: int = 5,
                 hdbscan_min_samples: int = 2,
                 random_state: int = 42,
                 embedding_field: str = "embedding"):
        """
        Initialize the clusterer with parameters for UMAP and HDBSCAN.
        These parameters will be adjusted dynamically based on dataset size.

        Args:
            umap_n_components: Target number of dimensions to reduce to
            umap_n_neighbors: Target number of neighbors for UMAP
            umap_min_dist: Minimum distance for UMAP
            hdbscan_min_cluster_size: Target minimum cluster size for HDBSCAN
            hdbscan_min_samples: Target minimum samples for HDBSCAN
            random_state: Random seed for reproducibility
            embedding_field: Field name in document.metadata containing the embedding
        """
        self.umap_n_components = umap_n_components
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_min_dist = umap_min_dist
        self.hdbscan_min_cluster_size = hdbscan_min_cluster_size
        self.hdbscan_min_samples = hdbscan_min_samples
        self.random_state = random_state
        self.embedding_field = embedding_field

    def _extract_embeddings(self, documents: List[Document]) -> np.ndarray:
        """Extract embeddings from documents."""
        embeddings = []
        for doc in documents:
            if self.embedding_field not in doc.metadata:
                raise ValueError(f"Document {doc.doc_id} missing embedding in metadata['{self.embedding_field}']")
            embeddings.append(doc.metadata[self.embedding_field])
        return np.array(embeddings)

    def _adjust_parameters(self, num_documents: int) -> Tuple[int, int, int, int]:
        """
        Adjust UMAP and HDBSCAN parameters based on dataset size.

        Args:
            num_documents: Number of documents in the dataset

        Returns:
            Tuple of (n_components, n_neighbors, min_cluster_size, min_samples)
        """
        # For UMAP n_components, use at most 1/2 of documents, minimum 2
        n_components = min(num_documents // 2, self.umap_n_components)
        n_components = max(2, n_components)  # At least 2 dimensions

        # For UMAP n_neighbors, use at most 1/3 of documents, minimum 2
        n_neighbors = min(num_documents // 3, self.umap_n_neighbors)
        n_neighbors = max(2, n_neighbors)  # At least 2 neighbors

        # For HDBSCAN min_cluster_size, use at most 1/4 of documents, minimum 2
        min_cluster_size = min(num_documents // 4, self.hdbscan_min_cluster_size)
        min_cluster_size = max(2, min_cluster_size)  # At least 2 per cluster

        # For HDBSCAN min_samples, use at most 1/5 of documents, minimum 1
        min_samples = min(num_documents // 5, self.hdbscan_min_samples)
        min_samples = max(1, min_samples)  # At least 1 sample

        return n_components, n_neighbors, min_cluster_size, min_samples

    def cluster_documents(self, documents: List[Document]) -> List[List[Document]]:
        """
        Cluster documents using UMAP and HDBSCAN.
        Parameters are automatically adjusted based on dataset size.

        Returns:
            List of document clusters (each cluster is a list of Documents)
        """
        if not documents:
            return []

        num_documents = len(documents)
        print(f"Clustering {num_documents} documents")

        # Extract embeddings
        embeddings = self._extract_embeddings(documents)

        # Dynamically adjust parameters based on dataset size
        n_components, n_neighbors, min_cluster_size, min_samples = self._adjust_parameters(num_documents)

        print("Adjusted parameters based on dataset size:")
        print(f"  - UMAP n_components: {n_components} (original: {self.umap_n_components})")
        print(f"  - UMAP n_neighbors: {n_neighbors} (original: {self.umap_n_neighbors})")
        print(f"  - HDBSCAN min_cluster_size: {min_cluster_size} (original: {self.hdbscan_min_cluster_size})")
        print(f"  - HDBSCAN min_samples: {min_samples} (original: {self.hdbscan_min_samples})")

        # For very small datasets, we might need to skip UMAP and use direct clustering
        if num_documents <= 5:
            print("Dataset too small for dimensionality reduction, clustering directly on embeddings")

            # Use a simple distance-based clustering
            from sklearn.cluster import AgglomerativeClustering

            # Use agglomerative clustering with cosine distance
            clusterer = AgglomerativeClustering(
                n_clusters=None,  # Let the algorithm decide based on distance
                distance_threshold=0.5,  # Adjust based on your embeddings
                linkage='average',
                affinity='cosine'
            )

            cluster_labels = clusterer.fit_predict(embeddings)

        else:
            # Dimension reduction with UMAP
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                min_dist=self.umap_min_dist,
                metric='cosine',
                random_state=self.random_state
            )
            reduced_embeddings = reducer.fit_transform(embeddings)

            # Clustering with HDBSCAN
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric='euclidean',
                cluster_selection_method='eom'
            )

            cluster_labels = clusterer.fit_predict(reduced_embeddings)

        # Group documents by cluster
        clusters: Dict[int, List[Document]] = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(documents[i])

        # Convert to list of lists
        result = [clusters[label] for label in sorted(clusters.keys())]

        # Print cluster sizes
        cluster_sizes = [len(cluster) for cluster in result]
        print(f"Created {len(result)} clusters with sizes: {cluster_sizes}")

        return result
