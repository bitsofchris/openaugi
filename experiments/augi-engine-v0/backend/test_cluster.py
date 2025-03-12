import os
import numpy as np
import pandas as pd
import plotly.express as px
from typing import List, Dict, Any

from llama_index.core.schema import Document
from clusterer import UMAPHDBSCANClusterer
import config

def test_clustering_small_dataset():
    """
    Test clustering on a small dataset and visualize the results.
    This script directly accesses LanceDB instead of using a get_all_atomic_notes method.
    """
    # Import your storage class
    # Make sure to change this import to match your actual module structure
    from storage import KnowledgeStore

    # Initialize components
    store = KnowledgeStore(config.LANCE_DB_PATH)

    # Access the database directly
    if store.atomic_notes_table not in store.db.table_names():
        print("Atomic notes table doesn't exist. Run extraction pipeline first.")
        return

    table = store.db.open_table(store.atomic_notes_table)

    # Load atomic notes directly
    print("Loading atomic notes directly from LanceDB...")
    all_data = table.to_pandas()

    print(f"Loaded {len(all_data)} atomic notes")

    # Convert to Document objects
    atomic_notes = []
    for _, row in all_data.iterrows():
        if "vector" not in row or row["vector"] is None:
            print(f"Skipping note {row.get('id', 'unknown')} without embedding")
            continue

        metadata = {}
        metadata["embedding"] = row["vector"]

        # Copy other metadata fields if available
        for field in ["idea_title", "source_doc_id", "source_doc_title"]:
            if field in row:
                metadata[field] = row[field]

        doc = Document(
            text=row.get("text", ""),
            metadata=metadata,
            doc_id=row.get("id", "")
        )
        atomic_notes.append(doc)

    print(f"Converted {len(atomic_notes)} notes to Document objects")

    if not atomic_notes:
        print("No atomic notes with embeddings found.")
        return

    # Initialize clusterer with dynamic parameter adjustment
    clusterer = UMAPHDBSCANClusterer()

    # Cluster the notes
    print("Clustering atomic notes...")
    clusters = clusterer.cluster_documents(atomic_notes)

    # For small datasets, we'll create a simple 2D scatter plot
    print("Creating visualization...")

    # Prepare data for Plotly
    viz_data = []

    # Assign cluster labels to each document
    doc_to_cluster = {}
    for i, cluster in enumerate(clusters):
        for doc in cluster:
            doc_to_cluster[doc.doc_id] = i

    # Extract embeddings for UMAP
    import umap

    embeddings = np.array([doc.metadata["embedding"] for doc in atomic_notes])

    # Use UMAP for dimensionality reduction (with adjusted parameters)
    n_neighbors = min(5, len(atomic_notes) - 1)  # Ensure n_neighbors is less than dataset size
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric='cosine',
        random_state=42
    )
    embeddings_2d = reducer.fit_transform(embeddings)

    # Create visualization data
    for i, doc in enumerate(atomic_notes):
        cluster_id = doc_to_cluster.get(doc.doc_id, -1)
        title = doc.metadata.get("idea_title", "Untitled")

        viz_data.append({
            "x": embeddings_2d[i, 0],
            "y": embeddings_2d[i, 1],
            "cluster": f"Cluster {cluster_id}",
            "title": title,
            "id": doc.doc_id,
            "text_preview": doc.text[:100] + "..." if len(doc.text) > 100 else doc.text
        })

    df = pd.DataFrame(viz_data)

    # Create the visualization
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="cluster",
        hover_data=["title", "id", "text_preview"],
        title="Atomic Notes Clusters"
    )

    # Create output directory if it doesn't exist
    os.makedirs("visualizations", exist_ok=True)

    # Save the visualization
    fig.write_html("visualizations/atomic_clusters.html")
    print("Visualization saved to visualizations/atomic_clusters.html")

    # For easier examination, save cluster info to CSV
    cluster_info = []
    for i, cluster in enumerate(clusters):
        for doc in cluster:
            title = doc.metadata.get("idea_title", "Untitled")
            cluster_info.append({
                "cluster_id": i,
                "doc_id": doc.doc_id,
                "title": title,
                "text": doc.text[:200] + "..." if len(doc.text) > 200 else doc.text
            })

    cluster_df = pd.DataFrame(cluster_info)
    cluster_df.to_csv("visualizations/cluster_contents.csv", index=False)
    print("Cluster contents saved to visualizations/cluster_contents.csv")

    return clusters

if __name__ == "__main__":
    clusters = test_clustering_small_dataset()