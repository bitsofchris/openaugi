import json
import re
import uuid
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from llama_index.core.schema import Document
from llama_index.llms.openai import OpenAI
from interfaces import Distiller, DistilledNote
from config import DOT_ENV_PATH


class ConceptDistiller(Distiller):
    """
    A distiller that focuses on identifying and elaborating key concepts within clusters.
    This approach preserves more of the nuance in the original notes.
    """

    def __init__(self, llm_model: str = "gpt-4o-mini-2024-07-18", min_cluster_size: int = 2):
        """
        Initialize the concept distiller with the specified LLM model.

        Args:
            llm_model: The model to use for distillation
            min_cluster_size: Minimum cluster size to process (smaller clusters are skipped)
        """
        load_dotenv(DOT_ENV_PATH)  # Load environment variables
        self.llm = OpenAI(model=llm_model, temperature=0.1)
        self.min_cluster_size = min_cluster_size

    def clean_llm_json_response(self, response_text: str) -> Dict[str, Any]:
        """Clean and parse JSON response from LLM."""
        # Pattern to match JSON content between triple backticks
        pattern = r"```(?:json)?\s*([\s\S]*?)```"

        # Try to find the pattern
        match = re.search(pattern, response_text)

        if match:
            # Extract the JSON content
            json_str = match.group(1)
            return json.loads(json_str)
        else:
            # If no backticks, try parsing the whole thing
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                # If that fails, raise an error
                raise ValueError("Could not extract valid JSON from the response")

    def identify_cluster_concepts(self, cluster: List[Document]) -> Dict[str, Any]:
        """
        Identify key concepts within a cluster.

        Args:
            cluster: List of related Documents

        Returns:
            Dictionary with concept analysis
        """
        # Prepare the context from all documents in the cluster
        note_contexts = []
        for i, doc in enumerate(cluster):
            title = doc.metadata.get("idea_title", f"Note {i+1}")
            note_contexts.append(f"NOTE {i+1}: {title}\n{doc.text}")

        notes_context = "\n\n".join(note_contexts)

        # Create the prompt for concept identification
        prompt = """
        Analyze these related notes and identify the key concepts they collectively address.

        NOTES:
        {notes_context}

        First, identify 3-5 central concepts discussed across these notes.
        For each concept:
        1. Give it a precise name
        2. Explain how different notes approach or elaborate on this concept
        3. Note any contradictions or different perspectives on this concept

        Return your analysis as JSON with this structure:
        {{
          "cluster_theme": "Overall theme of these notes",
          "concepts": [
            {{
              "name": "Concept name",
              "description": "Integrated explanation of this concept",
              "perspectives": [
                {{ "note_num": "1", "perspective": "How note 1 views this concept" }},
                {{ "note_num": "2", "perspective": "How note 2 views this concept" }}
              ],
              "contradictions": "Any contradictory views (if applicable)"
            }}
          ]
        }}

        Ensure you return valid JSON.
        """

        response = self.llm.complete(prompt.format(notes_context=notes_context))

        try:
            return self.clean_llm_json_response(response.text)
        except Exception as e:
            print(f"Failed to analyze cluster concepts: {str(e)}")
            return {
                "cluster_theme": "Analysis failed",
                "concepts": []
            }

    def create_concept_synthesis(self,
                                 cluster: List[Document],
                                 concept_analysis: Dict[str, Any]) -> Optional[DistilledNote]:
        """
        Create a concept-oriented synthesis based on concept analysis.

        Args:
            cluster: List of related Documents
            concept_analysis: The concept analysis from identify_cluster_concepts

        Returns:
            DistilledNote object or None if processing fails
        """
        # Create a structured synthesis from the concept analysis
        concepts = concept_analysis.get("concepts", [])
        if not concepts:
            return None

        # Build the content
        title = concept_analysis.get("cluster_theme", "Concept Synthesis")

        content_parts = [f"# {title}\n"]

        # Add key concepts sections
        for concept in concepts:
            concept_name = concept.get("name", "Unnamed Concept")
            content_parts.append(f"## {concept_name}")

            # Add the integrated description
            content_parts.append(concept.get("description", ""))

            # Add perspectives section if available
            perspectives = concept.get("perspectives", [])
            if perspectives:
                content_parts.append("\n### Different Perspectives")
                for perspective in perspectives:
                    note_num = perspective.get("note_num", "")
                    view = perspective.get("perspective", "")
                    if note_num and view:
                        content_parts.append(f"- **Note {note_num}**: {view}")

            # Add contradictions if any
            contradictions = concept.get("contradictions", "")
            if contradictions and contradictions.lower() not in ["none", "n/a", "not applicable"]:
                content_parts.append("\n### Tensions or Contradictions")
                content_parts.append(contradictions)

            content_parts.append("\n")  # Add space between concepts

        # Join all content
        full_content = "\n".join(content_parts)

        # Create metadata
        metadata = {
            "title": title,
            "key_concepts": [c.get("name", "") for c in concepts if "name" in c],
            "sibling_notes": [],  # Can be filled in later if needed
            "is_user_edited": False
        }

        # Create Document
        doc = Document(
            text=full_content,
            metadata=metadata,
            doc_id=str(uuid.uuid4())
        )

        # Get source IDs
        source_ids = [d.doc_id for d in cluster]

        return DistilledNote(doc, source_ids)

    def distill_single_cluster(self, cluster: List[Document]) -> Optional[DistilledNote]:
        """
        Distill a single cluster using the concept-oriented approach.

        Args:
            cluster: List of related Documents

        Returns:
            DistilledNote object or None if processing fails
        """
        if len(cluster) < self.min_cluster_size:
            print(f"Skipping cluster with only {len(cluster)} notes (min size: {self.min_cluster_size})")
            return None

        # First identify core concepts
        concept_analysis = self.identify_cluster_concepts(cluster)

        # Then create synthesis based on concepts
        return self.create_concept_synthesis(cluster, concept_analysis)

    def distill_knowledge(self, clusters: List[List[Document]]) -> List[DistilledNote]:
        """
        Distill/summarize clusters into higher-level ideas using concept-oriented approach.

        Args:
            clusters: List of document clusters (each cluster is a list of Documents)

        Returns:
            List of DistilledNote objects
        """
        distilled_notes = []

        for i, cluster in enumerate(clusters):
            print(f"Distilling cluster {i+1}/{len(clusters)} with {len(cluster)} notes (concept approach)...")

            note = self.distill_single_cluster(cluster)
            if note:
                distilled_notes.append(note)

        print(f"Created {len(distilled_notes)} concept-based distilled notes from {len(clusters)} clusters")
        return distilled_notes

