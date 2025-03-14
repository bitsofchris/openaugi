import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import umap
from typing import List, Optional, Dict, Tuple
from llama_index.core.schema import Document


class KnowledgeMapVisualizer:
    """Enhanced visualizer that creates a mindmap-like view of knowledge."""

    def __init__(self, output_dir: str = "visualizations"):
        """Initialize the visualizer."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def create_2d_projection(self, documents: List[Document]) -> np.ndarray:
        """Create a 2D projection of document embeddings."""
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
            metric="cosine",
            random_state=42,
        )

        return reducer.fit_transform(embeddings)

    def _get_short_description(self, doc: Document) -> str:
        """Get a short 1-3 word description from document title."""
        # Get title from metadata
        title = doc.metadata.get("title", "")
        if not title:
            title = doc.metadata.get("idea_title", "")

        if not title:
            return "Untitled"

        # Take first 3 words max
        words = title.split()
        if len(words) <= 3:
            return title
        else:
            return " ".join(words[:3]) + "..."

    def _is_distilled_note(self, doc: Document) -> bool:
        """Check if a document is a distilled note."""
        # Logic to determine if this is a distilled note
        # Look for the presence of source atomic note IDs
        if "atomic_note_ids" in doc.metadata and doc.metadata["atomic_note_ids"]:
            return True
        # Or check if it's in the clean notes
        if "is_clean_note" in doc.metadata and doc.metadata["is_clean_note"]:
            return True
        return False

    def _prepare_document_data(
        self,
        atomic_notes: List[Document],
        distilled_notes: Optional[List[Document]] = None,
    ) -> Tuple[List[Document], List[int], Dict[int, List[int]]]:
        """
        Prepare document data for visualization.

        Args:
            atomic_notes: List of atomic notes
            distilled_notes: Optional list of distilled notes

        Returns:
            Tuple containing:
            - all_docs: Combined list of all documents
            - distilled_indices: Indices of distilled notes
            - source_map: Mapping of distilled note indices to source note indices
        """
        all_docs = atomic_notes.copy()
        distilled_indices = []
        source_map = {}  # Maps distilled note index to source note indices

        if not distilled_notes:
            return all_docs, distilled_indices, source_map

        start_idx = len(all_docs)
        for i, note in enumerate(distilled_notes):
            distilled_idx = start_idx + i
            distilled_indices.append(distilled_idx)

            # Find source notes
            source_ids = note.metadata.get("atomic_note_ids", [])
            # Handle pandas/numpy array types
            if hasattr(source_ids, "__iter__") and not isinstance(
                source_ids, (list, str)
            ):
                source_ids = [str(id) for id in source_ids if id]

            if isinstance(source_ids, list) and len(source_ids) > 0:
                # Find indices of source notes
                source_indices = []
                for j, atomic in enumerate(atomic_notes):
                    if atomic.doc_id in source_ids:
                        source_indices.append(j)

                source_map[distilled_idx] = source_indices

        all_docs.extend(distilled_notes)
        return all_docs, distilled_indices, source_map

    def _create_node_data(
        self,
        all_docs: List[Document],
        distilled_indices: List[int],
        embeddings_2d: np.ndarray,
    ) -> pd.DataFrame:
        """
        Create dataframe with node information for visualization.

        Args:
            all_docs: Combined list of all documents
            distilled_indices: Indices of distilled notes
            embeddings_2d: 2D projection of document embeddings

        Returns:
            DataFrame with node visualization data
        """
        viz_data = []

        for i, doc in enumerate(all_docs):
            is_distilled = i in distilled_indices

            # Get title and description
            if is_distilled:
                title = doc.metadata.get("title", "Untitled Distilled Note")
                description = self._get_short_description(doc)
                node_type = "distilled"
            else:
                title = doc.metadata.get("idea_title", "Untitled")
                description = title  # For atomic notes, description is the title
                node_type = "atomic"

            viz_data.append(
                {
                    "x": embeddings_2d[i, 0],
                    "y": embeddings_2d[i, 1],
                    "node_type": node_type,
                    "title": title,
                    "description": description,
                    "id": doc.doc_id,
                    "text_preview": (
                        doc.text[:100] + "..." if len(doc.text) > 100 else doc.text
                    ),
                }
            )

        return pd.DataFrame(viz_data)

    def _add_atomic_nodes(self, fig: go.Figure, df: pd.DataFrame) -> go.Figure:
        """
        Add atomic notes to the figure.

        Args:
            fig: Plotly figure
            df: DataFrame with node data

        Returns:
            Updated figure
        """
        atomic_df = df[df["node_type"] == "atomic"]
        if atomic_df.empty:
            return fig

        atomic_scatter = go.Scatter(
            x=atomic_df["x"],
            y=atomic_df["y"],
            mode="markers",
            marker=dict(
                size=10,
                color="rgba(0, 120, 220, 0.7)",
                line=dict(width=1, color="DarkSlateGrey"),
            ),
            text=atomic_df["title"],
            hovertemplate="<b>%{text}</b><br>ID: %{customdata[0]}<br>%{customdata[1]}",
            customdata=list(zip(atomic_df["id"], atomic_df["text_preview"])),
            name="Atomic Ideas",
        )
        fig.add_trace(atomic_scatter)
        return fig

    def _add_distilled_nodes(self, fig: go.Figure, df: pd.DataFrame) -> go.Figure:
        """
        Add distilled notes to the figure.

        Args:
            fig: Plotly figure
            df: DataFrame with node data

        Returns:
            Updated figure
        """
        distilled_df = df[df["node_type"] == "distilled"]
        if distilled_df.empty:
            return fig

        distilled_scatter = go.Scatter(
            x=distilled_df["x"],
            y=distilled_df["y"],
            mode="markers+text",
            marker=dict(
                size=20,
                color="rgba(220, 50, 50, 0.7)",
                symbol="diamond",
                line=dict(width=2, color="DarkSlateGrey"),
            ),
            text=distilled_df["description"],
            textposition="top center",
            hovertemplate="<b>%{customdata[0]}</b><br>ID: %{customdata[1]}<br>%{customdata[2]}",
            customdata=list(
                zip(
                    distilled_df["title"],
                    distilled_df["id"],
                    distilled_df["text_preview"],
                )
            ),
            name="Distilled Knowledge",
        )
        fig.add_trace(distilled_scatter)
        return fig

    def _add_connections(
        self,
        fig: go.Figure,
        source_map: Dict[int, List[int]],
        embeddings_2d: np.ndarray,
    ) -> go.Figure:
        """
        Add connection lines between distilled notes and their source notes.

        Args:
            fig: Plotly figure
            source_map: Mapping of distilled note indices to source note indices
            embeddings_2d: 2D projection of document embeddings

        Returns:
            Updated figure
        """
        for distilled_idx, source_indices in source_map.items():
            distilled_x = embeddings_2d[distilled_idx, 0]
            distilled_y = embeddings_2d[distilled_idx, 1]

            for source_idx in source_indices:
                source_x = embeddings_2d[source_idx, 0]
                source_y = embeddings_2d[source_idx, 1]

                # Create a line connecting distilled note to source
                connection = go.Scatter(
                    x=[distilled_x, source_x],
                    y=[distilled_y, source_y],
                    mode="lines",
                    line=dict(color="rgba(180, 180, 180, 0.4)", width=1, dash="dot"),
                    hoverinfo="none",
                    showlegend=False,
                )
                fig.add_trace(connection)
        return fig

    def create_knowledge_map(
        self,
        atomic_notes: List[Document],
        distilled_notes: Optional[List[Document]] = None,
        include_connections: bool = True,
        show_atomic_notes: bool = True,
    ) -> str:
        """
        Create an interactive knowledge map visualization.

        Args:
            atomic_notes: List of atomic notes
            distilled_notes: Optional list of distilled notes
            include_connections: Whether to draw connections from distilled notes to source notes
            show_atomic_notes: Whether to display atomic notes (set to False to show only distilled notes)

        Returns:
            Path to the saved visualization
        """
        # Prepare document data
        all_docs, distilled_indices, source_map = self._prepare_document_data(
            atomic_notes, distilled_notes
        )

        # Create 2D projection for all documents
        embeddings_2d = self.create_2d_projection(all_docs)

        # Create node data
        df = self._create_node_data(all_docs, distilled_indices, embeddings_2d)

        # Create figure
        fig = go.Figure()

        # Add atomic notes if requested
        if show_atomic_notes:
            fig = self._add_atomic_nodes(fig, df)

        # Add distilled notes if available
        if distilled_notes:
            fig = self._add_distilled_nodes(fig, df)

            # Add connections if requested and if atomic notes are visible
            if include_connections and show_atomic_notes:
                fig = self._add_connections(fig, source_map, embeddings_2d)

        # Set figure layout
        title_text = "Knowledge Map"
        if not show_atomic_notes and distilled_notes:
            title_text = "Distilled Knowledge Map"

        fig.update_layout(
            title=title_text,
            plot_bgcolor="white",
            legend_title_text="Node Type",
            height=800,
            width=1100,
            hoverlabel=dict(bgcolor="white", font_size=12),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )

        # Save outputs
        output_path = os.path.join(self.output_dir, "knowledge_map.html")
        fig.write_html(output_path)
        df.to_csv(os.path.join(self.output_dir, "knowledge_map.csv"), index=False)

        return output_path

    def visualize_knowledge_base(self, store, include_connections: bool = True) -> str:
        """
        Visualize the entire knowledge base from a KnowledgeStore.

        Args:
            store: KnowledgeStore instance
            include_connections: Whether to draw connections between notes

        Returns:
            Path to the saved visualization
        """
        # Load atomic notes
        atomic_notes = store.get_all_atomic_notes()
        print(f"Loaded {len(atomic_notes)} atomic notes")

        # Load clean/distilled notes if available
        distilled_notes = []
        if hasattr(store, "get_all_clean_notes"):
            distilled_notes = store.get_all_clean_notes()
            print(f"Loaded {len(distilled_notes)} distilled notes")

        # Create visualization
        return self.create_knowledge_map(
            atomic_notes=atomic_notes,
            distilled_notes=distilled_notes,
            include_connections=include_connections,
        )


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
            metric="cosine",
            random_state=42,
        )

        return reducer.fit_transform(embeddings)

    def visualize_clusters(
        self, clusters: List[List[Document]], title: str = "Document Clusters"
    ) -> str:
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

            viz_data.append(
                {
                    "x": embeddings_2d[i, 0],
                    "y": embeddings_2d[i, 1],
                    "cluster": f"Cluster {cluster_id}",
                    "title": title_field,
                    "id": doc.doc_id,
                    "text_preview": (
                        doc.text[:100] + "..." if len(doc.text) > 100 else doc.text
                    ),
                }
            )

        df = pd.DataFrame(viz_data)

        # Create interactive plot
        fig = px.scatter(
            df,
            x="x",
            y="y",
            color="cluster",
            hover_data=["title", "id", "text_preview"],
            title=title,
        )

        fig.update_traces(
            marker=dict(size=10, opacity=0.7), selector=dict(mode="markers")
        )

        fig.update_layout(
            plot_bgcolor="white", legend_title_text="Cluster", height=800, width=1000
        )

        # Save visualization
        output_path = os.path.join(
            self.output_dir, f"{title.lower().replace(' ', '_')}.html"
        )
        fig.write_html(output_path)

        # Save CSV with cluster contents
        csv_path = os.path.join(
            self.output_dir, f"{title.lower().replace(' ', '_')}.csv"
        )
        df.to_csv(csv_path, index=False)

        return output_path
