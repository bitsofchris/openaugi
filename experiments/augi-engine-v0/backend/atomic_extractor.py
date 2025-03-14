import json
import re
import os
import logging
from datetime import datetime
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
from llama_index.core.schema import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from interfaces import AtomicExtractor
from config import DOT_ENV_PATH


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("atomic_extractor")

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)


class SimpleLLMAtomicExtractor(AtomicExtractor):
    def __init__(self, llm_model: str = "gpt-4o-mini-2024-07-18"):
        """
        Initialize the extractor with the specified LLM model.

        Args:
            llm_model: The model to use for extraction.
        """
        load_dotenv(DOT_ENV_PATH)  # Load environment variables
        self.llm = OpenAI(model=llm_model, temperature=0.1)
        self.text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=100)

        # Create error log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.error_log_path = f"logs/json_errors_{timestamp}.log"

    def log_json_error(self, error_type: str, response_text: str, error_details: str = None):
        """
        Log JSON parsing errors to a file for later inspection.

        Args:
            error_type: Type of error (e.g., 'extraction', 'deduplication')
            response_text: The raw LLM response that failed to parse
            error_details: Additional error details
        """
        with open(self.error_log_path, "a") as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"ERROR TYPE: {error_type}\n")
            f.write(f"TIMESTAMP: {datetime.now().isoformat()}\n")
            if error_details:
                f.write(f"DETAILS: {error_details}\n")
            f.write(f"RAW RESPONSE:\n{response_text}\n")
            f.write(f"{'='*80}\n")

        logger.warning(f"JSON parsing error in {error_type}. Details logged to {self.error_log_path}")

    def clean_llm_json_response(self, response_text: str) -> Optional[List[Dict[str, Any]]]:
        """
        Clean and parse JSON from LLM response.
        Instead of raising exceptions, returns None if parsing fails.

        Args:
            response_text: Raw response text from LLM

        Returns:
            Parsed JSON object or None if parsing fails
        """
        # Pattern to match JSON content between triple backticks
        pattern = r"```(?:json)?\s*([\s\S]*?)```"

        # Try to find the pattern
        match = re.search(pattern, response_text)

        try:
            if match:
                # Extract the JSON content
                json_str = match.group(1)
                return json.loads(json_str)
            else:
                # If no backticks, try parsing the whole thing
                return json.loads(response_text)
        except json.JSONDecodeError as e:
            # Log the error and return None instead of raising exception
            self.log_json_error("json_parsing", response_text, str(e))
            return None

    def split_into_sections(self, text: str) -> List[str]:
        """
        Split document into manageable sections.

        Args:
            text: The full document text

        Returns:
            List of text sections
        """
        # First attempt to detect natural sections (headings, paragraphs, etc.)
        section_markers = [
            r"\n+\s*#+\s+",  # Markdown headings
            r"\n+\s*Section\s+\d+",  # Explicit section labels
            r"\n+\s*Topic\s*:|\n+\s*Subject\s*:",  # Topic markers
            r"\n{2,}",  # Multiple newlines (paragraph breaks)
        ]

        for marker in section_markers:
            sections = re.split(marker, text)
            if len(sections) > 1:
                # Found natural sections, clean them up
                return [s.strip() for s in sections if s.strip()]

        # If no natural sections are found, fall back to chunking
        chunks = self.text_splitter.split_text(text)
        return chunks

    def generate_global_summary(self, document: Document) -> str:
        """
        Generate a global summary of the document.

        Args:
            document: The Document object

        Returns:
            Summary text
        """
        prompt = """
        Create a concise summary that captures:
        1. The main topics discussed in this document
        2. Key points for each topic
        3. Any important relationships between concepts

        Keep the summary focused and highlight only the most important information.

        DOCUMENT:
        {text}

        SUMMARY:
        """

        response = self.llm.complete(prompt.format(text=document.text))
        return response.text

    def extract_section_ideas(
        self, section: str, global_summary: str
    ) -> List[Dict[str, Any]]:
        """
        Extract atomic ideas from a section, with reference to the global summary.

        Args:
            section: The section text
            global_summary: The global summary for context

        Returns:
            List of atomic ideas with metadata or empty list if extraction fails
        """
        standard_prompt = """
        You are extracting distinct atomic ideas from a section of text.

        Here is the global context of the entire document:
        {global_summary}

        Now analyze this specific section and extract 3-5 distinct atomic ideas.
        For each idea:
        1. Provide a concise title (5-7 words)
        2. Provide a clear description (1-2 sentences)
        3. Identify any related ideas mentioned in the global summary

        Format your response as a JSON list:
        [
          {{
            "title": "Idea title",
            "description": "Idea description",
            "links": ["related idea title", "related idea title"]
          }}
        ]

        SECTION TEXT:
        {section}

        ATOMIC IDEAS (JSON format):
        """

        try:
            response = self.llm.complete(
                standard_prompt.format(global_summary=global_summary, section=section)
            )

            # Try to parse the JSON response
            ideas = self.clean_llm_json_response(response.text)

            if ideas and len(ideas) > 0:
                return ideas
            else:
                # If parsing failed, log it and return fallback
                logger.warning(f"Failed to extract ideas from section: {section[:100]}...")
                return self._create_fallback_idea(section)

        except Exception as e:
            # Catch any other exceptions during processing
            logger.error(f"Error in section idea extraction: {str(e)}")
            self.log_json_error("section_extraction", f"Section: {section[:200]}...", str(e))
            return self._create_fallback_idea(section)

    def _create_fallback_idea(self, section: str) -> List[Dict[str, Any]]:
        """
        Create a fallback idea when extraction fails.

        Args:
            section: The section text

        Returns:
            List with one minimal idea
        """
        return [
            {
                "title": "Section content",
                "description": section[:100] + "..." if len(section) > 100 else section,
                "links": [],
            }
        ]

    def deduplicate_ideas(
        self, all_ideas: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Deduplicate similar ideas across sections.

        Args:
            all_ideas: All extracted ideas

        Returns:
            Deduplicated list of ideas
        """
        if not all_ideas:
            return []

        # Prepare all ideas as context
        ideas_context = "\n".join(
            [
                f"IDEA {i+1}: {idea['title']} - {idea['description']}"
                for i, idea in enumerate(all_ideas)
            ]
        )

        prompt = """
        You are deduplicating similar ideas extracted from a document.

        Here are all the extracted ideas:
        {ideas_context}

        For each idea, determine if it is a duplicate or very similar to another idea.
        Group similar ideas and provide a merged representation that captures all important details.

        Return your answer as a JSON list of deduplicated ideas:

        [
          {{
            "title": "Merged idea title",
            "description": "Merged idea description",
            "links": ["related concept 1", "related concept 2"],
            "source_ideas": [original idea indices that were merged]
          }}
        ]

        Ensure you return valid JSON.
        """

        try:
            response = self.llm.complete(prompt.format(ideas_context=ideas_context))
            result = self.clean_llm_json_response(response.text)

            if result:
                return result
            else:
                # If deduplication fails, log it and return original ideas
                logger.warning("Deduplication failed, returning original ideas")
                return all_ideas

        except Exception as e:
            # Catch any other exceptions during deduplication
            logger.error(f"Error in deduplication: {str(e)}")
            self.log_json_error("deduplication", str(e))
            return all_ideas  # Return original ideas if deduplication fails

    def extract_atomic_ideas(self, documents: List[Document], store=None) -> List[Document]:
        """
        Extract atomic ideas from documents, linking back to source docs.
        Skip documents that have already been processed if store is provided.

        Args:
            documents: List of source Documents
            store: Optional AtomicIdeaStore to check for already processed documents

        Returns:
            List of new Documents, each representing an atomic idea
        """
        all_atomic_documents = []
        documents_to_process = []

        # First filter out already processed documents if store is provided
        if store:
            for doc in documents:
                if not store.has_processed_document(doc.doc_id):
                    documents_to_process.append(doc)

            if len(documents_to_process) < len(documents):
                logger.info(f"Skipping {len(documents) - len(documents_to_process)} already processed documents")
        else:
            documents_to_process = documents

        # If no documents need processing, return empty list
        if not documents_to_process:
            logger.info("No new documents to process")
            return []

        logger.info(f"Processing {len(documents_to_process)} documents")

        for i, doc in enumerate(documents_to_process):
            try:
                logger.info(f"Processing document {i+1}/{len(documents_to_process)}: {doc.doc_id}")

                # Generate a global summary
                global_summary = self.generate_global_summary(doc)

                # Split into sections
                sections = self.split_into_sections(doc.text)
                logger.info(f"Document split into {len(sections)} sections")

                # Extract ideas from each section
                all_ideas = []
                for j, section in enumerate(sections):
                    try:
                        section_ideas = self.extract_section_ideas(section, global_summary) or []
                        all_ideas.extend(section_ideas)
                        logger.info(f"Section {j+1}: Extracted {len(section_ideas)} ideas")
                    except Exception as e:
                        logger.error(f"Error processing section {j+1}: {str(e)}")
                        self.log_json_error(f"section_processing_{doc.doc_id}_{j}", section[:200], str(e))
                        # Continue with next section instead of failing

                # Deduplicate ideas
                original_idea_count = len(all_ideas)
                deduplicated_ideas = self.deduplicate_ideas(all_ideas)
                logger.info(f"Deduplicated from {original_idea_count} to {len(deduplicated_ideas)} ideas")

                # Convert each idea to a Document
                for idea in deduplicated_ideas:
                    # Create metadata that links back to the source document
                    metadata = {
                        "source_doc_id": doc.doc_id,
                        "source_doc_title": doc.metadata.get("title", ""),
                        "source_doc_path": doc.metadata.get("file_path", ""),
                        "idea_title": idea["title"],
                        "links": idea.get("links", []),
                        "is_atomic_idea": True
                    }

                    # Create a new Document for this atomic idea
                    atomic_doc = Document(
                        text=idea["description"],
                        metadata=metadata
                    )

                    all_atomic_documents.append(atomic_doc)

            except Exception as e:
                # Catch any errors at the document level
                logger.error(f"Error processing document {doc.doc_id}: {str(e)}")
                self.log_json_error(f"document_processing_{doc.doc_id}",
                                    f"Document ID: {doc.doc_id}\nTitle: {doc.metadata.get('title', 'Unknown')}",
                                    str(e))
                # Continue with next document instead of failing

        logger.info(f"Extracted {len(all_atomic_documents)} atomic ideas from {len(documents_to_process)} documents")
        return all_atomic_documents