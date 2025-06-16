# Parse events from events.md
# Fetch from vault notes created 1 day before and after each event
# Prompt LLM with event + surrounding notes

import os
from datetime import datetime, timedelta
from typing import List, Dict
from llama_index.readers.obsidian import ObsidianReader
from llama_index.core.schema import Document
import logging
from pathlib import Path
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
import json


# Set up logging
def setup_logging():
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Create a log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"event_analysis_{timestamp}.log"

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),  # This will also print to console
        ],
    )
    return log_file


class EventParser:
    def __init__(self, events_file: str):
        self.events_file = events_file

    def parse_events(self) -> List[Dict]:
        """Parse events from the markdown file."""
        events = []
        current_event = None

        with open(self.events_file, "r") as f:
            for line in f:
                # Check for event header (### YYYY-MM-DD)
                if line.startswith("### "):
                    if current_event:
                        events.append(current_event)

                    date_str = line.strip().replace("### ", "")
                    current_event = {
                        "date": datetime.strptime(date_str, "%Y-%m-%d"),
                        "notes": [],
                        "context": [],
                        "previous": [],
                        "other_journals": [],
                    }
                elif current_event:
                    # Parse different sections
                    if line.startswith("Notes:"):
                        current_event["notes"].append(
                            line.replace("Notes:", "").strip()
                        )
                    elif line.startswith("Context:"):
                        current_event["context"].append(
                            line.replace("Context:", "").strip()
                        )
                    elif line.startswith("Previous:"):
                        current_event["previous"].append(
                            line.replace("Previous:", "").strip()
                        )
                    elif line.startswith("Other Journals:"):
                        current_event["other_journals"].append(
                            line.replace("Other Journals:", "").strip()
                        )

        if current_event:
            events.append(current_event)

        return events


class VaultContextFetcher:
    def __init__(self, vault_path: str):
        self.vault_path = vault_path
        self.reader = ObsidianReader(vault_path)

    def get_context_notes(self, event_date: datetime) -> List[Document]:
        """Get notes from 1 day before and after the event date."""
        start_date = event_date - timedelta(days=1)
        end_date = event_date + timedelta(days=1)

        # Load all documents
        all_docs = self.reader.load_data()
        logging.info(f"Loaded {len(all_docs)} documents")

        # Filter documents by file timestamps
        context_docs = []
        for doc in all_docs:
            file_path = os.path.join(
                doc.metadata["folder_path"], doc.metadata["file_name"]
            )
            try:
                # Get file stats
                stats = os.stat(file_path)
                # Created time not working on my vault copy b/c it's not accurate
                created_time = datetime.fromtimestamp(stats.st_ctime)
                # modified_time = datetime.fromtimestamp(stats.st_mtime)

                # Check if either timestamp falls within our window
                if start_date <= created_time <= end_date:
                    # or (start_date <= modified_time <= end_date):
                    logging.info(f"Adding document: {file_path}")
                    logging.info(f"Text: {doc.text}")
                    context_docs.append(doc)
            except (OSError, FileNotFoundError):
                # Skip files that can't be accessed
                continue

        return context_docs


class EventAnalyzer:
    def __init__(self, model: str = "gpt-4.1-2025-04-14"):
        """Initialize the analyzer with the specified LLM model."""
        load_dotenv()  # Load environment variables
        self.llm = OpenAI(model=model, temperature=0.1)

        # Create analysis results directory
        self.results_dir = Path("analysis_results")
        self.results_dir.mkdir(exist_ok=True)

        # Create timestamp for this analysis run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.results_dir / self.timestamp
        self.run_dir.mkdir(exist_ok=True)

        # Store individual note analyses
        self.note_analyses = []

    def analyze_note(
        self, event: Dict, context_note: Document, prompt_template: str
    ) -> Dict:
        """Analyze a single context note in relation to the event."""
        # Prepare the context
        context = {
            "event_date": event["date"].strftime("%Y-%m-%d"),
            "event_notes": "\n".join(event["notes"]),
            "event_context": "\n".join(event["context"]),
            "event_previous": "\n".join(event["previous"]),
            "event_other_journals": "\n".join(event["other_journals"]),
            "context_note_path": context_note.metadata.get("file_path", "Unknown"),
            "context_note_content": context_note.text,
        }

        # Format the prompt
        prompt = prompt_template.format(**context)

        try:
            # Get LLM response
            response = self.llm.complete(prompt)

            # Create analysis result
            analysis = {
                "event_date": event["date"].isoformat(),
                "context_note": context_note.metadata.get("file_path", "Unknown"),
                "analysis": response.text,
                "timestamp": datetime.now().isoformat(),
            }

            # Store the analysis
            self.note_analyses.append(analysis)

            return analysis

        except Exception as e:
            logging.error(
                f"Error analyzing note {context_note.metadata.get('file_path')}: {str(e)}"
            )
            return None

    def generate_meta_analysis(self, event: Dict, meta_prompt_template: str) -> Dict:
        """Generate a meta-analysis of all note analyses for an event."""
        if not self.note_analyses:
            return None

        # Prepare the context
        context = {
            "event_date": event["date"].strftime("%Y-%m-%d"),
            "event_notes": "\n".join(event["notes"]),
            "event_context": "\n".join(event["context"]),
            "event_previous": "\n".join(event["previous"]),
            "event_other_journals": "\n".join(event["other_journals"]),
            "note_analyses": json.dumps(self.note_analyses, indent=2),
        }

        # Format the prompt
        prompt = meta_prompt_template.format(**context)

        try:
            # Get LLM response
            response = self.llm.complete(prompt)

            # Create meta-analysis result
            meta_analysis = {
                "event_date": event["date"].isoformat(),
                "meta_analysis": response.text,
                "timestamp": datetime.now().isoformat(),
                "note_analyses": self.note_analyses,
            }

            # Save to file
            output_file = (
                self.run_dir / f"meta_analysis_{event['date'].strftime('%Y%m%d')}.json"
            )
            with open(output_file, "w") as f:
                json.dump(meta_analysis, f, indent=2)

            return meta_analysis

        except Exception as e:
            logging.error(
                f"Error generating meta-analysis for event {event['date']}: {str(e)}"
            )
            return None


def main():
    # Set up logging
    log_file = setup_logging()
    logging.info(f"Starting event analysis. Log file: {log_file}")

    # Initialize components
    events_file = "data/events.md"
    vault_path = "/Users/chris/zk-for-testing"

    # Define prompts
    note_analysis_prompt = """
    Analyze this note in relation to the event that occurred on {event_date}.
    
    Event Details:
    Notes: {event_notes}
    Context: {event_context}
    Previous Events: {event_previous}
    Other Journals: {event_other_journals}
    
    Context Note ({context_note_path}):
    {context_note_content}
    
    Please analyze:
    1. How does this note relate to the event?
    2. What insights can we draw from this note about the event?
    3. Are there any patterns or connections that emerge?
    
    Provide your analysis in a clear, structured format.
    """

    meta_analysis_prompt = """
    Analyze all the individual note analyses for the event on {event_date}.
    
    Event Details:
    Notes: {event_notes}
    Context: {event_context}
    Previous Events: {event_previous}
    Other Journals: {event_other_journals}
    
    Individual Note Analyses:
    {note_analyses}
    
    Please provide a meta-analysis that:
    1. Identifies common themes across all analyses
    2. Highlights any significant patterns or trends
    3. Draws broader insights about the event and its context
    4. Suggests any potential connections to other events or patterns
    
    Provide your meta-analysis in a clear, structured format.
    """

    # Parse events
    event_parser = EventParser(events_file)
    events = event_parser.parse_events()

    logging.info(f"\n=== Found {len(events)} events ===\n")

    # Initialize vault fetcher and analyzer
    vault_fetcher = VaultContextFetcher(vault_path)
    analyzer = EventAnalyzer()

    # Process each event
    for event in events:
        logging.info(f"\n{'='*50}")
        logging.info(f"Event from {event['date'].strftime('%Y-%m-%d')}")
        logging.info(f"{'='*50}")

        # Print event details
        logging.info("\nEvent Details:")
        logging.info("-" * 20)
        if event["notes"]:
            logging.info("\nNotes:")
            for note in event["notes"]:
                logging.info(f"  - {note}")

        if event["context"]:
            logging.info("\nContext:")
            for ctx in event["context"]:
                logging.info(f"  - {ctx}")

        if event["previous"]:
            logging.info("\nPrevious:")
            for prev in event["previous"]:
                logging.info(f"  - {prev}")

        if event["other_journals"]:
            logging.info("\nOther Journals:")
            for journal in event["other_journals"]:
                logging.info(f"  - {journal}")

        # Get context notes
        context_notes = vault_fetcher.get_context_notes(event["date"])
        logging.info(f"\nFound {len(context_notes)} context notes")

        # TODO
        return

        # Analyze each context note
        for doc in context_notes:
            logging.info(
                f"\nAnalyzing note: {doc.metadata.get('file_path', 'Unknown')}"
            )
            analysis = analyzer.analyze_note(event, doc, note_analysis_prompt)
            if analysis:
                logging.info("Analysis completed successfully")

        # Generate meta-analysis for this event
        logging.info("\nGenerating meta-analysis...")
        meta_analysis = analyzer.generate_meta_analysis(event, meta_analysis_prompt)
        if meta_analysis:
            logging.info("Meta-analysis completed successfully")
            logging.info("\nMeta-analysis results:")
            logging.info(meta_analysis["meta_analysis"])

        logging.info("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    main()
