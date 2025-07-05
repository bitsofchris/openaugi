import streamlit as st
import pandas as pd
import plotly.express as px

# Configure the page
st.set_page_config(
    page_title="AI Conversation Clusters Explorer", page_icon="🧠", layout="wide"
)

st.title("🧠 AI Conversation Clusters Explorer")
st.markdown("Explore your conversation clusters with interactive visualizations")

# Define available CSV files
csv_files = {
    "K-means + UMAP": "../data/cluster_visualization_kmeans.csv",
    "HDBSCAN + UMAP": "../data/cluster_visualization_hdbscan.csv",
    "K-means + t-SNE": "../data/cluster_visualization_kmeans_tsne.csv",
    "HDBSCAN + t-SNE": "../data/cluster_visualization_hdbscan_tsne.csv",
}

# Sidebar controls
st.sidebar.header("📊 Visualization Options")
st.sidebar.markdown("---")

# File selection
selected_option = st.sidebar.selectbox(
    "Choose clustering method and projection:", options=list(csv_files.keys()), index=0
)


# Load selected file
@st.cache_data
def load_data(file_path):
    """Load and cache the CSV data"""
    try:
        df = pd.read_csv(file_path)
        # Convert date column to datetime for filtering
        df["formatted_date"] = pd.to_datetime(df["formatted_date"])
        return df, None
    except Exception as e:
        return None, str(e)


# Load the selected data
file_path = csv_files[selected_option]
df, error = load_data(file_path)

if error:
    st.error(f"❌ Error loading {selected_option}: {error}")
    st.stop()

if df is None:
    st.error(f"❌ Could not load {selected_option}")
    st.stop()

# Add date binning columns
df["date_week"] = df["formatted_date"].dt.to_period("W").astype(str)
df["date_month_year"] = df["formatted_date"].dt.to_period("M").astype(str)

# Convert thread names to integers for better performance
thread_name_mapping = {name: i for i, name in enumerate(df["thread_name"].unique())}
df["thread_name_int"] = df["thread_name"].map(thread_name_mapping)

# Display basic info
st.sidebar.success(f"✅ Loaded: {selected_option}")
st.sidebar.info(f"**Total Points:** {len(df):,}")
st.sidebar.info(f"**Unique Threads:** {len(thread_name_mapping)}")

# Filters section
st.sidebar.markdown("---")
st.sidebar.header("🔍 Filters")

# Date range filter
min_date = df["formatted_date"].min().date()
max_date = df["formatted_date"].max().date()

date_range = st.sidebar.date_input(
    "Date Range:", value=(min_date, max_date), min_value=min_date, max_value=max_date
)

# Role filter
available_roles = df["role"].unique()
selected_roles = st.sidebar.multiselect(
    "Role:", options=available_roles, default=available_roles
)

# Cluster size filter
cluster_sizes = df["cluster"].value_counts().sort_index()
min_cluster_size = st.sidebar.slider(
    "Minimum cluster size:", min_value=1, max_value=cluster_sizes.max(), value=1
)

# Search filter
search_term = st.sidebar.text_input(
    "Search in content:", placeholder="Enter keywords..."
)

# Color by dropdown
color_options = {
    "Cluster ID": "cluster",
    "Thread Name": "thread_name_int",
    "Week": "date_week",
    "Month-Year": "date_month_year",
}

color_by = st.sidebar.selectbox(
    "Color points by:", options=list(color_options.keys()), index=0
)

# Show note about thread name performance optimization
if color_by == "Thread Name":
    st.sidebar.caption(
        "💡 Thread names are converted to integers for better performance. Hover tooltips show actual thread names."
    )

# Cluster ID filter
available_clusters = sorted(df["cluster"].unique())
selected_clusters = st.sidebar.multiselect(
    "Filter by Cluster IDs:",
    options=available_clusters,
    default=[],
    placeholder="Select clusters to highlight/filter",
)

# Apply filters
filtered_df = df.copy()

# Date filter
if len(date_range) == 2:
    start_date, end_date = date_range
    filtered_df = filtered_df[
        (filtered_df["formatted_date"].dt.date >= start_date)
        & (filtered_df["formatted_date"].dt.date <= end_date)
    ]

# Role filter
if selected_roles:
    filtered_df = filtered_df[filtered_df["role"].isin(selected_roles)]

# Cluster size filter
valid_clusters = cluster_sizes[cluster_sizes >= min_cluster_size].index
filtered_df = filtered_df[filtered_df["cluster"].isin(valid_clusters)]

# Search filter
if search_term:
    mask = filtered_df["content"].str.contains(search_term, case=False, na=False)
    filtered_df = filtered_df[mask]

# Cluster filter
if selected_clusters:
    filtered_df = filtered_df[filtered_df["cluster"].isin(selected_clusters)]
    # Clear clicked cluster when filtering
    st.session_state.clicked_cluster_id = None

# Update sidebar info with filtered data
st.sidebar.info(f"**Filtered Points:** {len(filtered_df):,}")
st.sidebar.info(f"**Visible Clusters:** {filtered_df['cluster'].nunique()}")

# Main layout with columns
col1, col2 = st.columns([2, 1])

# Initialize session state for clicked cluster
if "clicked_cluster_id" not in st.session_state:
    # Default to the biggest cluster
    biggest_cluster = df["cluster"].value_counts().index[0]
    st.session_state.clicked_cluster_id = biggest_cluster

with col1:
    st.subheader(f"🎯 {selected_option} Visualization")

    if len(filtered_df) == 0:
        st.warning("⚠️ No data matches the current filters. Please adjust your filters.")
    else:
        # Prepare display dataframe with highlighting and short preview
        filtered_df_display = filtered_df.copy()

        # Add short preview for tooltip (25 chars)
        filtered_df_display["short_preview"] = (
            filtered_df_display["content"].str[:25] + "..."
        )

        # Add highlighting for selected clusters
        if selected_clusters:
            filtered_df_display["highlight"] = filtered_df_display["cluster"].isin(
                selected_clusters
            )
            filtered_df_display["opacity"] = filtered_df_display["highlight"].map(
                {True: 1.0, False: 0.3}
            )
            filtered_df_display["size"] = filtered_df_display["highlight"].map(
                {True: 2, False: 1}
            )
        else:
            filtered_df_display["opacity"] = 1.0
            filtered_df_display["size"] = 1.5

        # Get selected color column
        color_column = color_options[color_by]

        # Determine color scale based on data type
        if color_column in ["cluster", "thread_name_int"]:
            color_scale = "viridis"
            color_discrete = False
        else:
            color_scale = None
            color_discrete = True

        # Create the plotly scatter plot
        fig = px.scatter(
            filtered_df_display,
            x="x",
            y="y",
            color=color_column,
            opacity=filtered_df_display["opacity"],
            size=filtered_df_display["size"],
            hover_data={
                "thread_name": True,
                "formatted_date": True,
                "role": True,
                "content_preview": True,
                "cluster_label": True,
                "x": False,
                "y": False,
                "cluster": False,
            },
            color_continuous_scale=color_scale if not color_discrete else None,
            color_discrete_sequence=(
                px.colors.qualitative.Set3 if color_discrete else None
            ),
            title=f"{selected_option} - {len(filtered_df):,} points (Colored by {color_by})",
            labels={"x": "Dimension 1", "y": "Dimension 2", color_column: color_by},
            width=900,
            height=750,
        )

        # Add small border to all markers
        if selected_clusters:
            # Slightly thicker border for highlighted clusters
            fig.update_traces(
                marker=dict(
                    line=dict(width=0.8, color="white"),
                    sizemode="diameter",
                    sizeref=0.05,
                )
            )
        else:
            # Small border for all markers
            fig.update_traces(
                marker=dict(
                    line=dict(width=0.3, color="rgba(255,255,255,0.6)"),
                    sizemode="diameter",
                    sizeref=0.05,
                )
            )

        # Update hover template for better readability
        # Show actual thread name in tooltip even when coloring by thread_name_int
        tooltip_color_label = color_by
        tooltip_color_column = color_column
        if color_column == "thread_name_int":
            tooltip_color_label = "Thread Name"
            tooltip_color_column = "thread_name"

        fig.update_traces(
            hovertemplate="<b>%{customdata[0]}</b><br>"
            + "Date: %{customdata[1]}<br>"
            + "Role: %{customdata[2]}<br>"
            + f"{tooltip_color_label}: %{{customdata[6]}}<br>"
            + "Cluster ID: %{customdata[5]}<br>"
            + "Preview: %{customdata[4]}<br>"
            + "<extra></extra>",
            customdata=filtered_df_display[
                [
                    "thread_name",
                    "formatted_date",
                    "role",
                    "cluster_label",
                    "short_preview",
                    "cluster",
                    tooltip_color_column,
                ]
            ].values,
        )

        # Offset tooltip position
        fig.update_layout(
            hoverlabel=dict(
                bgcolor="white",
                bordercolor="gray",
                font_size=12,
                font_color="black",
                # Offset tooltip position
                align="left",
            )
        )

        # Display the plot with click and hover events
        plot_data = st.plotly_chart(
            fig, use_container_width=True, on_select="rerun", key="main_plot"
        )

with col2:
    st.subheader("📝 Content Details")

    if len(filtered_df) == 0:
        st.info("No data to display")
    elif plot_data and plot_data.selection and plot_data.selection.points:
        # Debug: Show the structure of the selection data
        st.write("Debug - Selection data:", plot_data.selection.points[0])

        # Get the clicked point index (try different possible key names)
        point_data = plot_data.selection.points[0]
        point_idx = None

        # Try common key names
        for key in ["point_index", "pointIndex", "index", "row"]:
            if key in point_data:
                point_idx = point_data[key]
                break

        # If still None, try to get any numeric value
        if point_idx is None:
            for key, value in point_data.items():
                if isinstance(value, (int, float)) and value >= 0:
                    point_idx = int(value)
                    break

        # Fallback to 0 if nothing works
        if point_idx is None:
            point_idx = 0

        # Map back to original dataframe index
        selected_row = filtered_df.iloc[point_idx]

        # Store clicked cluster ID for cluster details display
        st.session_state.clicked_cluster_id = selected_row["cluster"]

        st.success("✅ Point selected!")

        # Display details
        st.write(f"**Thread:** {selected_row['thread_name']}")
        st.write(
            f"**Date:** {selected_row['formatted_date'].strftime('%Y-%m-%d %H:%M')}"
        )
        st.write(f"**Role:** {selected_row['role']}")
        st.write(f"**Cluster:** {selected_row['cluster_label']}")

        st.markdown("**Full Content:**")
        st.text_area(
            "Content", value=selected_row["content"], height=400, disabled=True
        )
    else:
        st.info("👆 Click on a point in the plot to see full content")
        st.markdown(
            """
        **Instructions:**
        1. Select clustering method from sidebar
        2. Use filters to narrow down data
        3. Hover over points for quick preview  
        4. Click on any point to see full content
        """
        )

# Statistics section
st.markdown("---")
st.subheader("📊 Dataset Statistics")

stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)

with stats_col1:
    st.metric("Filtered Points", f"{len(filtered_df):,}")
    st.caption(f"Total: {len(df):,}")

with stats_col2:
    st.metric("Visible Clusters", filtered_df["cluster"].nunique())
    st.caption(f"Total: {df['cluster'].nunique()}")

with stats_col3:
    st.metric("Unique Threads", filtered_df["thread_name"].nunique())
    st.caption(f"Total: {df['thread_name'].nunique()}")

with stats_col4:
    if len(filtered_df) > 0:
        role_counts = filtered_df["role"].value_counts()
        st.metric("Human Messages", f"{role_counts.get('human', 0):,}")
        st.caption(f"Assistant: {role_counts.get('assistant', 0):,}")
    else:
        st.metric("Human Messages", "0")
        st.caption("Assistant: 0")

# Show top clusters
if len(filtered_df) > 0:
    st.markdown("---")
    st.subheader("🔥 Top Clusters")

    cluster_stats = filtered_df["cluster"].value_counts().head(10)
    cluster_info = []

    for cluster_id, count in cluster_stats.items():
        cluster_data = filtered_df[filtered_df["cluster"] == cluster_id]
        sample_content = cluster_data["content_preview"].iloc[0]
        cluster_info.append(
            {
                "Cluster": cluster_id,
                "Size": count,
                "Sample Content": (
                    sample_content[:100] + "..."
                    if len(sample_content) > 100
                    else sample_content
                ),
            }
        )

    cluster_df = pd.DataFrame(cluster_info)

    # Make table interactive
    st.markdown(
        "**💡 Tip:** Copy cluster IDs from the table above and paste into the Cluster ID filter to highlight/filter specific clusters"
    )

    # Display with selection
    event = st.dataframe(
        cluster_df,
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row",
    )

    # Handle cluster selection from table
    if event.selection["rows"]:
        selected_row_idx = event.selection["rows"][0]
        selected_cluster_id = cluster_df.iloc[selected_row_idx]["Cluster"]
        st.info(
            f"Selected Cluster {selected_cluster_id} - Add it to the sidebar filter to highlight!"
        )

# Show detailed cluster content
clusters_to_show = []

# Determine which clusters to show in detail
if selected_clusters:
    clusters_to_show = selected_clusters
elif st.session_state.clicked_cluster_id is not None:
    clusters_to_show = [st.session_state.clicked_cluster_id]

# Display cluster details if we have data
if len(filtered_df) > 0:
    st.markdown("---")
    st.subheader("🔍 Cluster Details")

    # Add cluster selection modal
    available_clusters_for_details = sorted(filtered_df["cluster"].unique())

    # Default to clicked cluster, filtered cluster, or biggest cluster
    if (
        st.session_state.clicked_cluster_id is not None
        and st.session_state.clicked_cluster_id in available_clusters_for_details
    ):
        default_cluster = st.session_state.clicked_cluster_id
    elif clusters_to_show and clusters_to_show[0] in available_clusters_for_details:
        default_cluster = clusters_to_show[0]
    else:
        # Default to biggest cluster in current filtered data
        default_cluster = filtered_df["cluster"].value_counts().index[0]

    # Cluster selection input
    col_a, col_b = st.columns([3, 1])
    with col_a:
        selected_detail_cluster = st.selectbox(
            "Select cluster to view details:",
            options=available_clusters_for_details,
            index=(
                available_clusters_for_details.index(default_cluster)
                if default_cluster in available_clusters_for_details
                else 0
            ),
            key="cluster_detail_selector",
        )

    with col_b:
        if st.button("🎯 Show Current"):
            if st.session_state.clicked_cluster_id is not None:
                st.session_state.cluster_detail_selector = (
                    st.session_state.clicked_cluster_id
                )
                st.rerun()

    # Show helpful info
    st.caption(
        "💡 Select any cluster above, or click a point in the plot to view its cluster details."
    )

    # Override clusters_to_show with selected cluster
    clusters_to_show = [selected_detail_cluster]

    for cluster_id in clusters_to_show:
        cluster_data = filtered_df[filtered_df["cluster"] == cluster_id]

        if len(cluster_data) > 0:
            st.markdown(f"**Cluster {cluster_id}** ({len(cluster_data)} points)")

            # Create display table with thread title and content
            detail_rows = []
            for idx, row in cluster_data.iterrows():
                detail_rows.append(
                    {
                        "Thread": row["thread_name"],
                        "Date": row["formatted_date"].strftime("%Y-%m-%d %H:%M"),
                        "Role": row["role"],
                        "Content": (
                            row["content"][:200] + "..."
                            if len(row["content"]) > 200
                            else row["content"]
                        ),
                    }
                )

            detail_df = pd.DataFrame(detail_rows)

            # Display with expander to save space
            with st.expander(
                f"Show {len(cluster_data)} messages in Cluster {cluster_id}"
            ):
                st.dataframe(detail_df, use_container_width=True, height=400)

            st.markdown("")  # Add spacing
