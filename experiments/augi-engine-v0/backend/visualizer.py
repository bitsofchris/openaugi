import os
import numpy as np
import pandas as pd
import plotly.express as px
import umap
from typing import List
from llama_index.core.schema import Document


class ClusterVisualizer:
    """Modular component for visualizing clusters."""

    def __init__(self, output_dir: str = "visualizations"):
        """
        Initialize the visualizer.

        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def create_2d_projection(self, documents: List[Document]) -> np.ndarray:
        """
        Create a 2D projection of document embeddings.

        Args:
            documents: List of documents with embeddings

        Returns:
            2D array of coordinates
        """
        # Extract embeddings
        embeddings = []
        for doc in documents:
            if "embedding" not in doc.metadata:
                raise ValueError(f"Document {doc.doc_id} missing embedding")
            embeddings.append(doc.metadata["embedding"])

        embeddings = np.array(embeddings)

        # Create 2D projection with UMAP
        n_neighbors = min(15, len(documents) - 1)
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=0.1,
            metric='cosine',
            random_state=42
        )

        return reducer.fit_transform(embeddings)

    def visualize_clusters(self, clusters: List[List[Document]], title: str = "Document Clusters") -> str:
        """
        Create interactive visualization of clusters.

        Args:
            clusters: List of document clusters
            title: Title for the visualization

        Returns:
            Path to the saved visualization
        """
        # Flatten documents
        all_docs = []
        for cluster in clusters:
            all_docs.extend(cluster)

        # Create document to cluster mapping
        doc_to_cluster = {}
        for i, cluster in enumerate(clusters):
            for doc in cluster:
                doc_to_cluster[doc.doc_id] = i

        # Create 2D projection
        embeddings_2d = self.create_2d_projection(all_docs)

        # Create visualization data
        viz_data = []
        for i, doc in enumerate(all_docs):
            cluster_id = doc_to_cluster.get(doc.doc_id, -1)

            # Get document title, preferring idea_title if available
            title_field = doc.metadata.get("idea_title", "")
            if not title_field and "title" in doc.metadata:
                title_field = doc.metadata["title"]
            if not title_field:
                title_field = f"Document {doc.doc_id[-6:]}"

            viz_data.append({
                "x": embeddings_2d[i, 0],
                "y": embeddings_2d[i, 1],
                "cluster": f"Cluster {cluster_id}",
                "title": title_field,
                "id": doc.doc_id,
                "text_preview": doc.text[:100] + "..." if len(doc.text) > 100 else doc.text
            })

        df = pd.DataFrame(viz_data)

        # Create interactive plot
        fig = px.scatter(
            df,
            x="x",
            y="y",
            color="cluster",
            hover_data=["title", "id", "text_preview"],
            title=title
        )

        fig.update_traces(
            marker=dict(size=10, opacity=0.7),
            selector=dict(mode='markers')
        )

        fig.update_layout(
            plot_bgcolor='white',
            legend_title_text='Cluster',
            height=800,
            width=1000
        )

        # Save visualization
        output_path = os.path.join(self.output_dir, f"{title.lower().replace(' ', '_')}.html")
        fig.write_html(output_path)

        # Save CSV with cluster contents
        csv_path = os.path.join(self.output_dir, f"{title.lower().replace(' ', '_')}.csv")
        df.to_csv(csv_path, index=False)

        return output_path