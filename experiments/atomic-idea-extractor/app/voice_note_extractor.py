from typing import List, Dict, Any
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI


class VoiceNoteExtractor:
    def __init__(self, llm_model: str = "gpt-4o-mini-2024-07-18"):
        """
        Initialize the extractor with the specified LLM model.

        Args:
            llm_model: The model to use.
        """
        self.llm = OpenAI(model=llm_model, temperature=0.1)
        self.text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=100)

    def load_transcript(self, file_path: str) -> str:
        """
        Load a transcript file.

        Args:
            file_path: Path to the transcript file

        Returns:
            The transcript text
        """
        with open(file_path, "r") as f:
            return f.read()

    def split_into_sections(self, text: str) -> List[str]:
        """
        Split transcript into sections.

        Args:
            text: The full transcript text

        Returns:
            List of text sections
        """
        # First attempt to detect natural sections (topics, timeframes, speakers)
        # This is more intelligent than simple chunking

        # Try to detect if there are natural section markers
        section_markers = [
            r"\n+\s*#+\s+",  # Markdown headings
            r"\n+\s*Section\s+\d+",  # Explicit section labels
            r"\n+\s*Topic\s*:|\n+\s*Subject\s*:",  # Topic markers
            r"\n+\s*\[\d+:\d+\]",  # Timestamp markers
            r"\n+\s*Speaker\s+\d+:|Speaker\s*:",  # Speaker changes
            r"\n{2,}",  # Multiple newlines (paragraph breaks)
        ]

        import re

        for marker in section_markers:
            sections = re.split(marker, text)
            if len(sections) > 1:
                # Found natural sections, clean them up
                return [s.strip() for s in sections if s.strip()]

        # If no natural sections are found, fall back to chunking
        chunks = self.text_splitter.split_text(text)
        return chunks

    def detect_document_type(self, text: str) -> str:
        """
        Detect the type of document to better inform the summarization strategy.

        Args:
            text: The document text

        Returns:
            Document type as string (meeting, brainstorm, lecture, note, etc.)
        """
        prompt = """
        Analyze this text and determine what type of document it is. 
        Choose ONE of the following categories that best describes it:
        - MEETING_NOTES (discussion between multiple people with action items)
        - BRAINSTORM (creative ideation with many possibilities discussed)
        - LECTURE (educational content with structured information)
        - PERSONAL_NOTES (stream of consciousness, personal thoughts)
        - INTERVIEW (question and answer format)
        - RESEARCH (analysis of findings, literature, or experiments)
        - OTHER (if none of the above fit well)
        
        Return ONLY the category name without explanation.
        
        TEXT:
        {text}
        
        CATEGORY:
        """

        sample_text = text[:2000] if len(text) > 2000 else text
        response = self.llm.complete(prompt.format(text=sample_text))
        return response.text.strip()

    def generate_global_summary(self, full_text: str) -> str:
        """
        Generate a global summary of the entire transcript.

        Args:
            full_text: The complete transcript

        Returns:
            Summary text
        """
        # First detect document type to customize prompting
        doc_type = self.detect_document_type(full_text)

        # Base prompt template
        base_prompt = """
        You are analyzing a {doc_type}. Create a concise summary that captures:
        1. The main topics discussed
        2. Key points for each topic
        3. {special_instruction}
        
        Keep the summary focused and highlight only the most important information.
        
        TEXT:
        {text}
        
        SUMMARY:
        """

        # Customize special instruction based on document type
        special_instructions = {
            "MEETING_NOTES": "Any decisions, action items, and who is responsible for them",
            "BRAINSTORM": "The main ideas generated and any potential next steps",
            "LECTURE": "The core concepts and their relationships",
            "PERSONAL_NOTES": "The key insights and any personal reflections",
            "INTERVIEW": "The main questions and answers, highlighting unique perspectives",
            "RESEARCH": "The key findings, methods, and implications",
            "OTHER": "Any decisions, action items, or conclusions",
        }

        special_instruction = special_instructions.get(
            doc_type, special_instructions["OTHER"]
        )

        # Create the final prompt
        prompt = base_prompt.format(
            doc_type=doc_type.lower().replace("_", " "),
            special_instruction=special_instruction,
            text=full_text,
        )

        response = self.llm.complete(prompt)
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
            List of atomic ideas with metadata
        """
        # First try with the standard prompt
        standard_prompt = """
        You are extracting distinct atomic ideas from a section of a voice note transcript.
        
        Here is the global context of the entire transcript:
        {global_summary}
        
        Now analyze this specific section and extract 3-5 distinct atomic ideas.
        For each idea:
        1. Provide a concise title (5-7 words)
        2. Provide a clear description (1-2 sentences)
        3. Identify any related ideas mentioned in the global summary
        4. If related ideas are found then create an Obsidian style link to that idea [[atomic idea title]]
        
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

        response = self.llm.complete(
            standard_prompt.format(global_summary=global_summary, section=section)
        )

        try:
            # Try to parse the JSON response
            import json

            ideas = json.loads(response.text)
            if ideas and len(ideas) > 0:
                return ideas
        except:
            # If parsing fails, try with a more structured prompt
            fallback_prompt = """
            Extract 3-5 atomic ideas from this text section. 
            The global context is: {global_summary}
            
            SECTION: {section}
            
            For each idea, follow this EXACT format with NO deviations:
            
            ```json
            [
              {{
                "title": "Short idea title",
                "description": "Brief description",
                "links": ["concept1", "concept2"]
              }},
              {{
                "title": "Another idea title",
                "description": "Another description",
                "links": []
              }}
            ]
            ```
            
            RESPOND ONLY WITH VALID JSON. Do not include any explanations before or after.
            """

            fallback_response = self.llm.complete(
                fallback_prompt.format(global_summary=global_summary, section=section)
            )

            try:
                # Extract JSON from the response - sometimes the model adds comments
                import re

                json_match = re.search(
                    r"```json\s*([\s\S]*?)\s*```", fallback_response.text
                )
                if json_match:
                    json_str = json_match.group(1)
                    return json.loads(json_str)
                else:
                    # Try to parse the whole response as JSON
                    return json.loads(fallback_response.text)
            except:
                print(f"Failed to parse JSON response in both attempts.")
                # Return a minimal valid structure
                return [
                    {
                        "title": "Section content",
                        "description": section[:100] + "...",
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
        You are deduplicating similar ideas extracted from a transcript.
        
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
        
        DEDUPLICATED IDEAS (JSON format):
        """

        response = self.llm.complete(prompt.format(ideas_context=ideas_context))

        try:
            import json

            return json.loads(response.text)
        except:
            print(f"Failed to parse JSON response in deduplication: {response.text}")
            return all_ideas  # Return original ideas if deduplication fails

    def identify_key_themes(self, ideas: List[Dict[str, Any]]) -> List[str]:
        """
        Identify overarching themes from all extracted ideas.

        Args:
            ideas: List of all extracted ideas

        Returns:
            List of key themes
        """
        if not ideas:
            return []

        # Prepare all ideas as context
        ideas_context = "\n".join(
            [
                f"IDEA {i+1}: {idea['title']} - {idea['description']}"
                for i, idea in enumerate(ideas)
            ]
        )

        prompt = """
        Analyze these ideas extracted from a document and identify 3-5 overarching themes that connect them.
        
        IDEAS:
        {ideas_context}
        
        Respond with ONLY a JSON array of theme names, for example:
        ["Theme 1", "Theme 2", "Theme 3"]
        """

        response = self.llm.complete(prompt.format(ideas_context=ideas_context))

        try:
            import json

            themes = json.loads(response.text)
            if isinstance(themes, list):
                return themes
            return []
        except:
            print(f"Failed to parse themes JSON: {response.text}")
            return []

    def process_transcript(self, transcript_text: str) -> Dict[str, Any]:
        """
        Process a transcript and extract atomic ideas.

        Args:
            transcript_text: The full transcript text

        Returns:
            Structured data with global summary and atomic ideas
        """
        # Detect document type
        doc_type = self.detect_document_type(transcript_text)

        # Generate global summary
        global_summary = self.generate_global_summary(transcript_text)

        # Split into sections
        sections = self.split_into_sections(transcript_text)

        # Extract ideas from each section
        all_ideas = []
        section_results = []

        for i, section in enumerate(sections):
            section_ideas = self.extract_section_ideas(section, global_summary)
            all_ideas.extend(section_ideas)

            section_results.append(
                {"section_id": i, "section_text": section, "ideas": section_ideas}
            )

        # Deduplicate ideas
        deduplicated_ideas = self.deduplicate_ideas(all_ideas)

        # Identify key themes
        themes = self.identify_key_themes(deduplicated_ideas)

        return {
            "document_type": doc_type,
            "global_summary": global_summary,
            "sections": section_results,
            "all_ideas": all_ideas,
            "deduplicated_ideas": deduplicated_ideas,
            "themes": themes,
        }


# Example usage
if __name__ == "__main__":
    extractor = VoiceNoteExtractor()
    transcript_text = extractor.load_transcript("sample.txt")
    results = extractor.process_transcript(transcript_text)

    # Save results to JSON file
    import json

    with open("extracted_ideas.json", "w") as f:
        json.dump(results, f, indent=2)

    print(
        f"Extracted {len(results['deduplicated_ideas'])} unique ideas from the transcript."
    )
