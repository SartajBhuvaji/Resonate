import json
import time
import pandas as pd
import sys
import os

# Ensuring the project's root directory is in the Python path for module importing
project_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# Importing the PineconeServerless class from the project's module
from src.pinecone.resonate_pinecone_functions import PineconeServerless


class TranscriptProcessor:
    """
    A class to process and upsert transcripts to Pinecone.

    Attributes:
        uuid_to_filename_dict (dict): A mapping from UUIDs to their respective transcript file names.
        pinecone (PineconeServerless): An instance of the PineconeServerless class for database operations.
    """

    def __init__(self):
        """
        Initializes the TranscriptProcessor with a predefined UUID to filename mapping and a PineconeServerless instance.
        """
        # Mapping UUIDs to their respective transcript file names
        self.uuid_to_filename_dict = {
            # UUID to filename mappings
            # Example: "uuid": "transcript_filename.csv",
        }
        # Initializing a PineconeServerless instance for database operations
        self.pinecone = PineconeServerless()

    def load_json_config(self, json_file_path=".//config/config.json"):
        """
        Loads a JSON configuration file.

        Parameters:
            json_file_path (str): The path to the JSON configuration file.

        Returns:
            dict: The data loaded from the JSON file.
        """
        with open(json_file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data

    def pinecone_init_upsert(
        self, df_transcript, meeting_title, meeting_summary, meeting_uuid
    ):
        """
        Initializes and performs an upsert operation to Pinecone with transcript data.

        Parameters:
            df_transcript (DataFrame): The transcript data as a pandas DataFrame.
            meeting_title (str): The title of the meeting.
            meeting_summary (str): The summary of the meeting.
            meeting_uuid (str): The UUID of the meeting.

        Exceptions:
            Catches and prints any exceptions raised during the upsert operation.
        """
        try:
            self.pinecone.pinecone_upsert(
                df_transcript,
                meeting_uuid=meeting_uuid,
                meeting_video_file=False,
                meeting_title=meeting_title,
                meeting_summary=meeting_summary,
            )
            # Wait for a short period to ensure the upsert operation completes
            time.sleep(5)
        except Exception as e:
            print("Error upserting transcript to Pinecone: ", e)

    def process_transcripts(self):
        """
        Processes and upserts all transcripts to Pinecone based on the UUID to filename mapping and the summary data.
        """
        summary_file = "./data/summaryFiles/abstract_summary_data.csv"
        df_summary = pd.read_csv(summary_file)
        # Creating a dictionary from the summaries DataFrame
        df_summary_dict = df_summary.set_index("uuid")["text"].to_dict()

        transcript_folder = "./data/transcriptFiles/"

        # Iterating over each UUID and its summary in the summary dictionary
        for uuid, summary in df_summary_dict.items():
            if uuid in self.uuid_to_filename_dict:
                filename = self.uuid_to_filename_dict[uuid]
                df_transcript = pd.read_csv(transcript_folder + filename)
                meeting_title = filename.replace(".csv", "")
                meeting_uuid = uuid

                # Performing the upsert operation for each transcript
                self.pinecone_init_upsert(
                    df_transcript, meeting_title, summary, meeting_uuid
                )


if __name__ == "__main__":
    processor = TranscriptProcessor()
    processor.process_transcripts()
