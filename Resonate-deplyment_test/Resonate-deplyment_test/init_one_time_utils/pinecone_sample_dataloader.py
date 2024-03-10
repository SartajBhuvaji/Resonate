# Give users at least 5 files to upload to Pinecone


import json
from src.pinecone.resonate_pinecone_functions import PineconeServerless
import time
import pandas as pd

uuid_to_filename_dict = {
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


pinecone = PineconeServerless()
def load_json_config(json_file_path="./config/config.json"):
    # Use a context manager to ensure the file is properly closed after opening
    with open(json_file_path, "r") as file:
        # Load the JSON data
        data = json.load(file)

    return data
def pinecone_init_upsert(
    df_transcript: pd.DataFrame,
    meeting_title: str,
    meeting_summary: str,
    meeting_uuid: str,
):
    try:

        pinecone = PineconeServerless()
        pinecone.pinecone_upsert(
            df_transcript,
            meeting_uuid=meeting_uuid,
            meeting_video_file=False,
            meeting_title=meeting_title,
            meeting_summary=meeting_summary,
        )
        time.sleep(5)
    except Exception as e:
        print("Error upserting transcript to Pinecone: ", e)


summaryFile = "./data/summaryFiles/abstract_summary_data.csv"

df_summary = pd.read_csv(summaryFile)
df_summary_dict_list = df_summary.to_dict(orient="records")
df_summary_dict = {}
for i in df_summary_dict_list:
    df_summary_dict[i["uuid"]] = i["text"]
display(df_summary_dict)

transcriptFolder = "./data/transcriptFiles/"

for i in df_summary_dict.keys():
    if i not in dnu_uuid:
        print(i, df_summary_dict[i])
        print(i, uuid_to_filename_dict[i])

        df_transcript = pd.read_csv(
            transcriptFolder + uuid_to_filename_dict[i],
        )
        meeting_title = uuid_to_filename_dict[i].replace(".csv", "")
        meeting_summary = df_summary_dict[i]
        meeting_uuid = i

        pinecone_init_upsert(
            df_transcript,
            meeting_title,
            meeting_summary,
            meeting_uuid,
        )