# Augi Engine v0 - Knowledge Distillation Pipeline

## Overview

Augi Engine v0 is an intelligent knowledge distillation system that transforms unstructured documents into a refined, interconnected knowledge base. The system extracts atomic ideas from documents, clusters related concepts, and distills them into higher-level insights while maintaining connections between original sources.

## System Architecture

### Core Components

The system follows a modular pipeline architecture with the following key components:

1. **Document Sources** - Load and prepare documents for processing
2. **Storage Layer** - Manage persistence and processing state tracking  
3. **Atomic Extractor** - Extract granular ideas from source documents
4. **Embedder** - Generate vector embeddings for semantic similarity
5. **Clusterer** - Group related atomic ideas based on semantic similarity
6. **Distiller/Selector** - Refine clusters into coherent concepts
7. **Visualizer** - Create visual knowledge maps and cluster representations

## Processing Pipeline

### Phase 1: Document Ingestion and Preprocessing

**ObsidianSource** (`sources.py`)
- Loads documents from Obsidian vault using LlamaIndex readers
- Generates unique, content-based document IDs using SHA-256 hashing
- Extracts tasks and processes metadata from markdown files
- Marks documents with `is_raw_document` metadata flag

**KnowledgeStore** (`storage.py`)
- Three-tier storage system using LanceDB vector database:
  - `raw_documents` - Original unprocessed files
  - `atomic_notes` - Extracted atomic ideas with source links
  - `clean_notes` - Distilled concepts with atomic source mapping
- Tracks processing state to avoid reprocessing unchanged documents
- Manages incremental updates and deduplication

### Phase 2: Atomic Idea Extraction

**SimpleLLMAtomicExtractor** (`atomic_extractor.py`)
- Uses GPT-4o-mini to extract 3-5 atomic ideas per document section
- Implements robust error handling for LLM JSON parsing failures
- Processes large documents in chunks using `SentenceSplitter`
- Generates global document summaries for contextual extraction
- Performs deduplication across extracted ideas
- Links atomic ideas back to source documents

### Phase 3: Semantic Processing

**LlamaIndexEmbedder** (`embedder.py`)
- Generates vector embeddings using OpenAI's `text-embedding-3-small`
- Handles large documents through chunking and embedding averaging
- Resilient processing with per-document error handling
- Stores embeddings in document metadata for downstream processing

### Phase 4: Clustering and Organization

**UMAPHDBSCANClusterer** (`clusterer.py`)
- Two-stage clustering approach:
  1. **UMAP**: Dimensionality reduction with cosine similarity metric
  2. **HDBSCAN**: Density-based clustering for natural groupings
- Dynamic parameter adjustment based on dataset size
- Fallback to agglomerative clustering for small datasets
- Saves cluster assignments to storage for persistence

### Phase 5: Knowledge Distillation

The system supports two distillation approaches:

#### Similarity-Based Distillation
**IntraClusterSimilarityFilter** (`selector.py`)
- Identifies highly similar documents within clusters (>85% cosine similarity)
- Creates smaller, more focused groups for targeted distillation
- Filters groups to ensure minimum viable size

#### Cluster-Based Distillation
**ConceptDistiller** (`distiller.py`)
- Processes entire clusters into synthesized concepts
- Uses LLM to identify key concepts and perspectives across notes
- Creates structured concept syntheses with:
  - Central theme identification
  - Multiple perspective integration
  - Contradiction and tension highlighting
- Maintains source traceability through `DistilledNote` objects

### Phase 6: Visualization and Output

**Visualizers** (`visualizer.py`)

**ClusterVisualizer**
- Creates scatter plots showing document clusters in 2D space
- Color-codes clusters for easy identification
- Displays cluster sizes and distribution

**KnowledgeMapVisualizer**
- Generates interactive mindmap-style visualizations
- Differentiates atomic notes (blue circles) from distilled concepts (larger red diamonds)
- Shows connections between distilled notes and their source atomic ideas
- Supports both atomic-only and full knowledge base views

## Data FlowRaw 
Documents → Atomic Extraction → Embedding → Clustering → Distillation → Visualization




## Configuration

**Config Options** (`config.py`)
- `DISTILLATION_METHOD`: Choose between "similarity" or "cluster" approaches
- `OBSIDIAN_VAULT_PATH`: Source document location
- `LANCE_DB_PATH`: Vector database storage path
- `SIMILARITY_THRESHOLD`: Cosine similarity threshold for grouping
- LLM model selection and embedding model configuration

## Key Features

### Incremental Processing
- Only processes new or changed documents
- Tracks processing state to avoid redundant work
- Supports iterative knowledge base refinement

### Robust Error Handling
- Comprehensive LLM response parsing with fallback strategies
- Per-document error isolation to prevent pipeline failures
- Detailed error logging for debugging

### Source Traceability
- Every distilled concept maintains links to source atomic ideas
- Atomic ideas link back to original documents
- Full provenance chain from raw documents to final concepts

### Flexible Distillation
- Multiple distillation strategies (similarity vs cluster-based)
- Configurable thresholds and parameters
- Concept-oriented synthesis preserving nuance and perspectives

## Usage

Run the complete pipeline:

```bash
python main.py
```

The main pipeline:
1. Loads new documents from the Obsidian vault
2. Extracts atomic ideas from unprocessed documents
3. Generates embeddings for all new content
4. Clusters atomic ideas by semantic similarity
5. Distills clusters into refined concepts
6. Generates visualizations of the knowledge base

## Output

- **Vector Database**: Persistent storage of all document levels with embeddings
- **Cluster Visualizations**: 2D scatter plots showing document groupings
- **Knowledge Maps**: Interactive visualizations connecting atomic ideas to distilled concepts
- **Distilled Notes**: Synthesized concepts with source traceability

The system creates a living knowledge base that grows incrementally and maintains rich connections between original sources and synthesized insights.