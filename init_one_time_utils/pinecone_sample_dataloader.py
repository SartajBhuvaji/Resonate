# Uploads data to pinecone
# Runner: python init_one_time_utils/pinecone_sample_dataloader.py
# Average Run Time: 35-40 min
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
             "52d105f8-1c80-4056-8253-732b9e2bec63": "office_relocation_1.csv",
                "9ed1fefa-db53-41fc-a21b-479b67e30073": "office_relocation_2.csv",
                "e993da88-0e17-4a35-ba9a-c03decca607b": "office_relocation_3.csv",
                "61d453f1-2852-48d9-a25a-b6e04c3c4908": "office_relocation_4.csv",
                "ba94585e-b0df-4633-bef2-a4f94f644c11": "Social_Media_-_Harmed_Teens.csv",
                "906c7694-0e33-4c8e-8f51-0365155fbb81": "Social_Media_-_Ruins_your_life.csv",
                "52d2dfe4-748b-4ecf-84fb-64be6ebcaeef": "ES2014a.Mix-Headset.csv",
                "1be8e439-45b3-4c97-9e4a-5c78c1a15e78": "ES2014b.Mix-Headset_1.csv",
                "a4b7b490-7b28-4744-85e5-d216f40ff52c": "ES2014b.Mix-Headset_2.csv",
                "b3821662-03f1-4349-8781-ba5f64439693": "ES2014c.Mix-Headset.csv",
                "95efa3c5-9770-4160-9f28-35350efb9f73": "Gitlab_Monthly_Release_Kickoff_1.csv",
                "85430eae-d466-4d63-9015-5835bbe71b90": "product_marketing_meeting.csv",
                "55d8afa8-a1bf-413c-a75c-b8c14da88d87": "Gitlab_Monthly_Release_Kickoff_2.csv",
                "15b7549d-4b3f-43b5-9507-85de435f1b4a": "2023-09-26_Architecture_Design_Workflow_New_Diffs_kickoff_call_1.csv",
                "875564dc-9954-41da-9084-ccf04ebffdb0": "2023-09-26_Architecture_Design_Workflow_New_Diffs_kickoff_call_2.csv",
                "72858a28-248d-4bef-af03-c62a3c285fbb": "2023-09-26_Architecture_Design_Workflow_New_Diffs_kickoff_call_3.csv",
                "4cbd0d4e-6cf9-4db4-bf15-f4f4e4d3d8d8": "2023-10-03-New_diffs_Architecture_Workflow.csv",
                "4badb5ba-ca92-4c3c-a7e9-0d49fc7a8137": "2023-10-10_New_diffs_architecture_workflow_weekly_EMEA_AMER_1.csv",
                "9c5aa3e4-b047-4f08-a838-9b665e251e4d": "2023-10-10_New_diffs_architecture_workflow_weekly_EMEA_AMER_2.csv",
                "d7c8e3b8-c6e0-4845-8669-f2f4ed1b8549": "2023-10-17_New_diffs_architecture_blueprint_1.csv",
                "876e67fa-314d-40e4-b942-21ca63e81995": "2023-10-17_New_diffs_architecture_blueprint_2.csv",
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
    
        for uuid, summary in df_summary_dict.items():
            if uuid in self.uuid_to_filename_dict:
                
                filename = self.uuid_to_filename_dict[uuid]
                df_transcript = pd.read_csv(transcript_folder + filename)
                meeting_title = filename.replace(".csv", "")
                meeting_uuid = uuid

                self.pinecone_init_upsert(
                    df_transcript, meeting_title, summary, meeting_uuid
                )
                time.sleep(20) # To prevent OPEN AI embedding limit error


if __name__ == "__main__":
    processor = TranscriptProcessor()
    processor.process_transcripts()